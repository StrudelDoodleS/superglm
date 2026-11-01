import Mathlib

/-!
# Issue 544, stage 2: the generalized gamma's uniform variable through `q = 0`

These statements are about real numbers. They certify the change of variables at the
head of `src/superglm/distributional/kernels/_gg_uniform_tail.py` (module docstring and
`_row_jets`'s comment "zeta = w G(q w)"):

* `lambda = exp(q w)`, `t = q w`, `g(t) = 2 (e^t - 1 - t)/t^2` with `g(0) = 1`;
* `zeta = w G(t)` with `G = sqrt g`, and `eta = q zeta`;
* `eta^2/2 = lambda - 1 - log lambda` with `sign eta = sign(lambda - 1)`, which is Temme's
  variable (DLMF 8.12.1), so for `q ≠ 0`, `zeta = eta/q`.

`g` and `G` are real analytic on all of `ℝ` and `G > 0`, so `zeta` is real analytic in
`(w, q)` jointly, through `q = 0`, where `zeta = w`. This is the step that removes the
apparent singularity of `eta/q` at `q = 0`.

Not covered: Temme's expansion itself (the series `S`, its coefficients' role, or its
truncation error), and floating-point accuracy.

Repository base: ff2ed52d (branch `claude/issue-544-gamma-bounds`). Date: 2026-11-01.
-/

namespace SuperGLM.Issue544.GG

open Real Filter Topology Set

/-- `g(t) = 2 (e^t - 1 - t)/t^2`, extended by its limit `g(0) = 1`. -/
noncomputable def g (t : ℝ) : ℝ := if t = 0 then 1 else 2 * (Real.exp t - 1 - t) / t ^ 2

/-- `G = sqrt g`, the code's `_G` series (`G(t) = sqrt(2 (e^t - 1 - t)/t^2)`). -/
noncomputable def G (t : ℝ) : ℝ := Real.sqrt (g t)

/-- `zeta(w, q) = w G(q w)`. -/
noncomputable def zeta (w q : ℝ) : ℝ := w * G (q * w)

/-- `eta(w, q) = q zeta(w, q)`. -/
noncomputable def eta (w q : ℝ) : ℝ := q * zeta w q

/-- Temme's variable as a function of `lambda` (DLMF 8.12.1):
`eta = sign(lambda - 1) sqrt(2 (lambda - 1 - log lambda))`. -/
noncomputable def temmeEta (lam : ℝ) : ℝ :=
  (SignType.sign (lam - 1) : ℝ) * Real.sqrt (2 * (lam - 1 - Real.log lam))

/-! ## `g` is analytic and positive -/

lemma rexp_hasFPowerSeriesAt :
    HasFPowerSeriesAt Real.exp (NormedSpace.expSeries ℝ ℝ) 0 := by
  rw [Real.exp_eq_exp_ℝ]
  exact NormedSpace.exp_hasFPowerSeriesAt_zero

lemma dslope_exp (t : ℝ) (ht : t ≠ 0) : dslope Real.exp 0 t = (Real.exp t - 1) / t := by
  rw [dslope_of_ne _ ht, slope_def_field]
  simp

lemma dslope_exp_zero : dslope Real.exp 0 0 = 1 := by
  rw [dslope_same, Real.deriv_exp, Real.exp_zero]

/-- `d2 = dslope (dslope exp 0) 0` has the power series `fslope (fslope expSeries)` at 0. -/
lemma d2_hasFPowerSeriesAt :
    HasFPowerSeriesAt (dslope (dslope Real.exp 0) 0)
      (NormedSpace.expSeries ℝ ℝ).fslope.fslope 0 :=
  rexp_hasFPowerSeriesAt.has_fpower_series_dslope_fslope.has_fpower_series_dslope_fslope

lemma d2_zero : dslope (dslope Real.exp 0) 0 0 = 1 / 2 := by
  have h := d2_hasFPowerSeriesAt.coeff_zero (fun _ => 1)
  rw [← h]
  have h2 : ((NormedSpace.expSeries ℝ ℝ).fslope.fslope 0) (fun _ => 1) =
      (NormedSpace.expSeries ℝ ℝ).fslope.fslope.coeff 0 := rfl
  rw [h2, FormalMultilinearSeries.coeff_fslope, FormalMultilinearSeries.coeff_fslope]
  change (NormedSpace.expSeries ℝ ℝ 2) (fun _ => (1 : ℝ)) = 1 / 2
  rw [NormedSpace.expSeries_apply_eq]
  norm_num [Nat.factorial]

/-- `g = 2 dslope (dslope exp 0) 0` everywhere, including `t = 0`. -/
lemma g_eq_dslope (t : ℝ) : g t = 2 * dslope (dslope Real.exp 0) 0 t := by
  by_cases ht : t = 0
  · subst ht
    rw [d2_zero]
    simp [g]
  · rw [dslope_of_ne _ ht, slope_def_field, dslope_exp t ht, dslope_exp_zero]
    simp only [g, ht, if_false, sub_zero]
    field_simp

/-- `t^2 g(t)/2 = e^t - 1 - t` for every real `t`. -/
theorem g_identity (t : ℝ) : t ^ 2 * g t / 2 = Real.exp t - 1 - t := by
  by_cases ht : t = 0
  · subst ht
    simp
  · simp only [g, ht, if_false]
    field_simp

/-- `g` is real analytic at every real point. -/
theorem g_analyticAt (t : ℝ) : AnalyticAt ℝ g t := by
  have hfun : g = fun x => 2 * dslope (dslope Real.exp 0) 0 x := funext g_eq_dslope
  by_cases ht : t = 0
  · subst ht
    rw [hfun]
    exact analyticAt_const.mul d2_hasFPowerSeriesAt.analyticAt
  · have hev : (fun x => 2 * (Real.exp x - 1 - x) / x ^ 2) =ᶠ[𝓝 t] g := by
      filter_upwards [isOpen_ne.mem_nhds ht] with x hx
      simp [g, hx]
    refine AnalyticAt.congr ?_ hev
    have hden : (t ^ 2 : ℝ) ≠ 0 := pow_ne_zero 2 ht
    have hnum : AnalyticAt ℝ (fun x => 2 * (Real.exp x - 1 - x)) t :=
      analyticAt_const.mul ((analyticAt_rexp.sub analyticAt_const).sub analyticAt_id)
    exact hnum.div (analyticAt_id.pow 2) hden

/-- `g` is `C^n` for every `n` (indeed analytic, `n = ω`). -/
theorem g_contDiff (n : WithTop ℕ∞) : ContDiff ℝ n g :=
  AnalyticOnNhd.contDiff fun t _ => g_analyticAt t

/-- `g(t) > 0` for every real `t`: `e^t > 1 + t` for `t ≠ 0` and `g(0) = 1`. -/
theorem g_pos (t : ℝ) : 0 < g t := by
  by_cases ht : t = 0
  · simp [g, ht]
  · simp only [g, ht, if_false]
    have h1 : t + 1 < Real.exp t := Real.add_one_lt_exp ht
    have h2 : 0 < t ^ 2 := by positivity
    apply div_pos _ h2
    linarith

/-! ## `G = sqrt g` and `zeta`, `eta` -/

lemma sqrt_analyticAt {y : ℝ} (hy : 0 < y) : AnalyticAt ℝ Real.sqrt y := by
  have hev : (fun x => Real.exp (Real.log x / 2)) =ᶠ[𝓝 y] Real.sqrt := by
    filter_upwards [isOpen_Ioi.mem_nhds hy] with x hx
    rw [Real.sqrt_eq_rpow, Real.rpow_def_of_pos hx]
    ring_nf
  refine AnalyticAt.congr ?_ hev
  have hl : AnalyticAt ℝ (fun x => Real.log x / 2) y := (analyticAt_log hy).div_const
  exact analyticAt_rexp.comp hl

/-- `G = sqrt g` is real analytic at every real point. -/
theorem G_analyticAt (t : ℝ) : AnalyticAt ℝ G t :=
  (sqrt_analyticAt (g_pos t)).comp (g_analyticAt t)

/-- `G(t) > 0` for every real `t`. -/
theorem G_pos (t : ℝ) : 0 < G t := Real.sqrt_pos.mpr (g_pos t)

theorem G_zero : G 0 = 1 := by simp [G, g]

/-- `zeta(w, q) = w G(q w)` is real analytic in `(w, q)` jointly, at every point,
including `q = 0`. -/
theorem zeta_analyticAt (p : ℝ × ℝ) : AnalyticAt ℝ (fun p : ℝ × ℝ => zeta p.1 p.2) p := by
  have hprod : AnalyticAt ℝ (fun p : ℝ × ℝ => p.2 * p.1) p := analyticAt_snd.mul analyticAt_fst
  have hG : AnalyticAt ℝ (fun p : ℝ × ℝ => G (p.2 * p.1)) p :=
    AnalyticAt.comp (g := G) (f := fun p : ℝ × ℝ => p.2 * p.1) (G_analyticAt (p.2 * p.1)) hprod
  exact analyticAt_fst.mul hG

/-- `zeta` is `C^n` in `(w, q)` for every `n`. -/
theorem zeta_contDiff (n : WithTop ℕ∞) : ContDiff ℝ n (fun p : ℝ × ℝ => zeta p.1 p.2) :=
  AnalyticOnNhd.contDiff fun p _ => zeta_analyticAt p

/-- At `q = 0` the uniform variable is the standardised point: `zeta(w, 0) = w`. -/
theorem zeta_q_zero (w : ℝ) : zeta w 0 = w := by simp [zeta, G_zero]

/-- `eta^2/2 = lambda - 1 - log lambda` with `lambda = exp(q w)`, for all real `w`, `q`. -/
theorem eta_sq (w q : ℝ) :
    eta w q ^ 2 / 2 = Real.exp (q * w) - 1 - Real.log (Real.exp (q * w)) := by
  rw [Real.log_exp, ← g_identity (q * w)]
  have h := Real.sq_sqrt (g_pos (q * w)).le
  simp only [eta, zeta, G]
  rw [show (q * (w * Real.sqrt (g (q * w)))) ^ 2 = (q * w) ^ 2 * Real.sqrt (g (q * w)) ^ 2 by ring,
    h]

/-- `lambda - 1 - log lambda = (q w)^2 g(q w)/2` (the identity that defines `g`). -/
theorem lambda_identity (w q : ℝ) :
    Real.exp (q * w) - 1 - Real.log (Real.exp (q * w)) = (q * w) ^ 2 * g (q * w) / 2 := by
  rw [Real.log_exp, g_identity]

/-- `eta = t G(t)` with `t = q w`. -/
theorem eta_eq (w q : ℝ) : eta w q = (q * w) * G (q * w) := by
  simp only [eta, zeta]
  ring

lemma sign_mul_pos {x y : ℝ} (hy : 0 < y) : SignType.sign (x * y) = SignType.sign x := by
  rw [_root_.sign_mul, _root_.sign_pos hy, mul_one]

lemma sign_exp_sub_one (t : ℝ) : SignType.sign (Real.exp t - 1) = SignType.sign t := by
  rcases lt_trichotomy t 0 with h | h | h
  · rw [_root_.sign_neg h, _root_.sign_neg]
    linarith [Real.exp_lt_one_iff.mpr h]
  · subst h
    simp
  · rw [_root_.sign_pos h, _root_.sign_pos]
    linarith [Real.one_lt_exp_iff.mpr h]

/-- `sign eta = sign(lambda - 1)` for all real `w`, `q`. -/
theorem sign_eta (w q : ℝ) :
    SignType.sign (eta w q) = SignType.sign (Real.exp (q * w) - 1) := by
  rw [eta_eq, sign_mul_pos (G_pos _), sign_exp_sub_one]

/-- `eta(w, q)` is Temme's variable of `lambda = exp(q w)`: the unique real number with
`eta^2/2 = lambda - 1 - log lambda` and the sign of `lambda - 1`. -/
theorem eta_eq_temme (w q : ℝ) : eta w q = temmeEta (Real.exp (q * w)) := by
  unfold temmeEta
  rw [← sign_eta, show 2 * (Real.exp (q * w) - 1 - Real.log (Real.exp (q * w))) =
      eta w q ^ 2 by linarith [eta_sq w q], Real.sqrt_sq_eq_abs]
  rcases lt_trichotomy (eta w q) 0 with h | h | h
  · rw [_root_.sign_neg h, abs_of_neg h]
    simp
  · rw [h]
    simp
  · rw [_root_.sign_pos h, abs_of_pos h]
    simp

/-- For `q ≠ 0`, `zeta = eta/q` with Temme's `eta`: `zeta` is the analytic extension of
`temmeEta(exp(q w))/q` through `q = 0`. -/
theorem zeta_eq_temme_div (w q : ℝ) (hq : q ≠ 0) :
    zeta w q = temmeEta (Real.exp (q * w)) / q := by
  rw [← eta_eq_temme, eta]
  field_simp

/-- `eta/(lambda - 1) = t G(t)/(e^t - 1)` for `t = q w ≠ 0`, a function of `t` alone:
the function whose Taylor coefficients in `eta` are the module's `_D`. -/
theorem eta_div_lambda (w q : ℝ) :
    eta w q / (Real.exp (q * w) - 1) = (q * w) * G (q * w) / (Real.exp (q * w) - 1) := by
  rw [eta_eq]

end SuperGLM.Issue544.GG

#print axioms SuperGLM.Issue544.GG.g_identity
#print axioms SuperGLM.Issue544.GG.g_analyticAt
#print axioms SuperGLM.Issue544.GG.g_contDiff
#print axioms SuperGLM.Issue544.GG.g_pos
#print axioms SuperGLM.Issue544.GG.G_analyticAt
#print axioms SuperGLM.Issue544.GG.G_pos
#print axioms SuperGLM.Issue544.GG.G_zero
#print axioms SuperGLM.Issue544.GG.zeta_analyticAt
#print axioms SuperGLM.Issue544.GG.zeta_contDiff
#print axioms SuperGLM.Issue544.GG.zeta_q_zero
#print axioms SuperGLM.Issue544.GG.eta_sq
#print axioms SuperGLM.Issue544.GG.lambda_identity
#print axioms SuperGLM.Issue544.GG.sign_eta
#print axioms SuperGLM.Issue544.GG.eta_eq_temme
#print axioms SuperGLM.Issue544.GG.zeta_eq_temme_div
