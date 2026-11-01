import Mathlib

/-!
# Issue 544, stage 2: the severity primitives and their derivative rules

These statements are about real numbers. They certify the real-valued definitions in
`src/superglm/distributional/kernels/_severity_primitives.py` and the SymPy `fdiff` rules
in `tools/generate_severity_families.py` (classes `softplus`, `expit`, `log1mexp`,
`invexpm1`, `gdev`, `gnorm`, `sres`, `psires`), on which every generated derivative rests:

* the elementary rules: `softplus' = expit`, `expit' = expit(z) expit(-z)`,
  `log1mexp' = invexpm1`, `invexpm1' = -r (1 + r)`, `gdev`'s two partials;
* the branch identities the numba bodies use (each branch is the same real function);
* Loader's series for `gdev` in `v = (m - y)/(m + y)`, the window `|v| < 1/3`, and the
  direct branch's bound `|y/m - 1| <= 3.3 gdev`;
* with `psi = Gamma'/Gamma` (equal to Mathlib's `Complex.digamma` on the positive reals):
  `gnorm' = -psires(0, .) = -sres(0, .)/a`, `psires(n)' = psires(n + 1)`,
  `sres(n)' = ((n + 1) sres(n) + sres(n + 1))/a`, the derivatives of `log` that
  `psires_0123` subtracts, and the shift recurrence `polygamma_0123` runs.

Not covered: the asymptotic (Stirling, DLMF 5.11.1-5.11.2) and origin (zeta) series that
`gnorm`, `psires_0123` and `sres_0123` evaluate, truncation of any series, and
floating-point accuracy. Mathlib (pinned rev) has no real digamma or polygamma, no
Stirling series with Bernoulli coefficients and no zeta series for `psi`.

Repository base: ff2ed52d (branch `claude/issue-544-gamma-bounds`). Date: 2026-11-01.
-/

namespace SuperGLM.Issue544.Primitives

open Real Set Filter Topology

/-! ## Elementary primitives -/

/-- `softplus(z) = log(1 + e^z)`. -/
noncomputable def softplus (z : ℝ) : ℝ := Real.log (1 + Real.exp z)

/-- `expit(z) = 1/(1 + e^-z)`. -/
noncomputable def expit (z : ℝ) : ℝ := 1 / (1 + Real.exp (-z))

/-- `log1mexp(t) = log(1 - e^-t)`, for `t > 0`. -/
noncomputable def log1mexp (t : ℝ) : ℝ := Real.log (1 - Real.exp (-t))

/-- `invexpm1(t) = 1/(e^t - 1)`. -/
noncomputable def invexpm1 (t : ℝ) : ℝ := 1 / (Real.exp t - 1)

/-- `gdev(y, m) = y/m - 1 - log(y/m)`, half the gamma unit deviance. -/
noncomputable def gdev (y m : ℝ) : ℝ := y / m - 1 - Real.log (y / m)

lemma one_add_exp_pos (z : ℝ) : 0 < 1 + Real.exp z := by positivity

/-- `expit(z) = e^z/(1 + e^z)`, the `z < 0` branch of `expit`. -/
theorem expit_branch (z : ℝ) : expit z = Real.exp z / (1 + Real.exp z) := by
  have h := Real.exp_pos z
  unfold expit
  rw [Real.exp_neg]
  field_simp
  ring

/-- `softplus(z) = z + log(1 + e^-z)`, the `z > 0` branch of `softplus`. -/
theorem softplus_branch (z : ℝ) : softplus z = z + Real.log (1 + Real.exp (-z)) := by
  unfold softplus
  have h1 : 1 + Real.exp z = Real.exp z * (1 + Real.exp (-z)) := by
    rw [mul_add, mul_one, ← Real.exp_add, add_neg_cancel, Real.exp_zero]
    ring
  rw [h1, Real.log_mul (Real.exp_pos z).ne' (one_add_exp_pos _).ne', Real.log_exp]

/-- `invexpm1(t) = e^-t/(1 - e^-t)` for `t ≠ 0`, the `t > 1` branch of `invexpm1`. -/
theorem invexpm1_branch {t : ℝ} (ht : t ≠ 0) :
    invexpm1 t = Real.exp (-t) / (1 - Real.exp (-t)) := by
  have h1 : Real.exp t ≠ 1 := fun h => ht (Real.exp_eq_one_iff t |>.mp h)
  have h2 : Real.exp t - 1 ≠ 0 := sub_ne_zero.mpr h1
  have h3 : Real.exp t ≠ 0 := (Real.exp_pos t).ne'
  have h4 : 1 - (Real.exp t)⁻¹ ≠ 0 := by
    intro h
    apply h1
    have := inv_eq_one.mp (by linarith : (Real.exp t)⁻¹ = 1)
    exact this
  unfold invexpm1
  rw [Real.exp_neg]
  field_simp

/-- `softplus' = expit` (the generator's `softplus.fdiff`). -/
theorem hasDerivAt_softplus (z : ℝ) : HasDerivAt softplus (expit z) z := by
  have h1 : HasDerivAt (fun x => 1 + Real.exp x) (Real.exp z) z :=
    (Real.hasDerivAt_exp z).const_add 1
  have h2 : HasDerivAt softplus (Real.exp z / (1 + Real.exp z)) z :=
    h1.log (one_add_exp_pos z).ne'
  rw [expit_branch]
  exact h2

/-- `expit' = expit(z) expit(-z)` (the generator's `expit.fdiff`). -/
theorem hasDerivAt_expit (z : ℝ) : HasDerivAt expit (expit z * expit (-z)) z := by
  have h1 : HasDerivAt (fun x => 1 + Real.exp (-x)) (-Real.exp (-z)) z := by
    simpa using ((hasDerivAt_neg z).exp).const_add 1
  have h2 := h1.inv (one_add_exp_pos (-z)).ne'
  have h3 : HasDerivAt expit (-(-Real.exp (-z)) / (1 + Real.exp (-z)) ^ 2) z := by
    have hfun : expit = (fun x => 1 + Real.exp (-x))⁻¹ := by
      funext x
      simp [expit]
    rw [hfun]
    exact h2
  convert h3 using 1
  have hz := Real.exp_pos z
  simp only [expit, neg_neg]
  rw [Real.exp_neg]
  field_simp
  ring

/-- `log1mexp' = invexpm1` for `t > 0` (the generator's `log1mexp.fdiff`). -/
theorem hasDerivAt_log1mexp {t : ℝ} (ht : 0 < t) : HasDerivAt log1mexp (invexpm1 t) t := by
  have hlt : Real.exp (-t) < 1 := Real.exp_lt_one_iff.mpr (by linarith)
  have h1 : HasDerivAt (fun x => 1 - Real.exp (-x)) (Real.exp (-t)) t := by
    simpa using ((hasDerivAt_neg t).exp).const_sub 1
  have h2 : HasDerivAt log1mexp (Real.exp (-t) / (1 - Real.exp (-t))) t :=
    h1.log (by linarith)
  rw [invexpm1_branch ht.ne']
  exact h2

/-- `invexpm1' = -r (1 + r)` with `r = invexpm1(t)`, for `t ≠ 0`
(the generator's `invexpm1.fdiff`). -/
theorem hasDerivAt_invexpm1 {t : ℝ} (ht : t ≠ 0) :
    HasDerivAt invexpm1 (-invexpm1 t * (1 + invexpm1 t)) t := by
  have hE : Real.exp t - 1 ≠ 0 := sub_ne_zero.mpr fun h => ht (Real.exp_eq_one_iff t |>.mp h)
  have h1 : HasDerivAt (fun x => Real.exp x - 1) (Real.exp t) t :=
    (Real.hasDerivAt_exp t).sub_const 1
  have h2 := h1.inv hE
  have h3 : HasDerivAt invexpm1 (-Real.exp t / (Real.exp t - 1) ^ 2) t := by
    have hfun : invexpm1 = (fun x => Real.exp x - 1)⁻¹ := by
      funext x
      simp [invexpm1]
    rw [hfun]
    exact h2
  convert h3 using 1
  unfold invexpm1
  field_simp
  ring

/-- `d gdev/dy = (y - m)/(y m)` for `y, m > 0` (the generator's `gdev.fdiff`, argument 1). -/
theorem hasDerivAt_gdev_y {y m : ℝ} (hy : 0 < y) (hm : 0 < m) :
    HasDerivAt (fun y => gdev y m) ((y - m) / (y * m)) y := by
  have h1 : HasDerivAt (fun y => y / m) (1 / m) y := (hasDerivAt_id y).div_const m
  have h2 : HasDerivAt (fun y => Real.log (y / m)) ((1 / m) / (y / m)) y :=
    h1.log (div_pos hy hm).ne'
  have h3 : HasDerivAt (fun y => gdev y m) (1 / m - (1 / m) / (y / m)) y :=
    (h1.sub_const 1).sub h2
  convert h3 using 1
  have hy' := hy.ne'
  have hm' := hm.ne'
  field_simp

/-- `d gdev/dm = (m - y)/m^2` for `y, m > 0` (the generator's `gdev.fdiff`, argument 2). -/
theorem hasDerivAt_gdev_m {y m : ℝ} (hy : 0 < y) (hm : 0 < m) :
    HasDerivAt (fun m => gdev y m) ((m - y) / m ^ 2) m := by
  have h1 : HasDerivAt (fun m => y / m) ((0 * m - y * 1) / m ^ 2) m :=
    (hasDerivAt_const m y).div (hasDerivAt_id' m) hm.ne'
  have h2 : HasDerivAt (fun m => Real.log (y / m)) (((0 * m - y * 1) / m ^ 2) / (y / m)) m :=
    h1.log (div_pos hy hm).ne'
  have h3 : HasDerivAt (fun m => gdev y m)
      ((0 * m - y * 1) / m ^ 2 - ((0 * m - y * 1) / m ^ 2) / (y / m)) m :=
    (h1.sub_const 1).sub h2
  convert h3 using 1
  have hy' := hy.ne'
  have hm' := hm.ne'
  field_simp
  ring

/-- The underflow branch of `gdev`: `log(y/m) = log y - log m` for `y, m > 0`. -/
theorem gdev_log_branch {y m : ℝ} (hy : 0 < y) (hm : 0 < m) :
    gdev y m = (y / m - 1) - (Real.log y - Real.log m) := by
  unfold gdev
  rw [Real.log_div hy.ne' hm.ne']

/-- The series window of `gdev`: `|v| < 1/3` exactly when `1/2 < y/m < 2`, for `y, m > 0`. -/
theorem gdev_window {y m : ℝ} (hy : 0 < y) (hm : 0 < m) :
    |(m - y) / (m + y)| < 1 / 3 ↔ 1 / 2 < y / m ∧ y / m < 2 := by
  have hs : 0 < m + y := by linarith
  rw [abs_div, abs_of_pos hs, div_lt_iff₀ hs, abs_lt, lt_div_iff₀ hm, div_lt_iff₀ hm]
  constructor
  · rintro ⟨h1, h2⟩
    constructor <;> linarith
  · rintro ⟨h1, h2⟩
    constructor <;> linarith

/-- Loader's series (*Fast and accurate computation of binomial probabilities*, 2000), the
`|v| < 1/3` branch of `gdev`: for `y, m > 0` and `v = (m - y)/(m + y)`,
`gdev(y, m) = (m - y) v/m + sum_{j >= 1} 2 v^(2j+1)/(2j+1)`. -/
theorem gdev_loader_series {y m : ℝ} (hy : 0 < y) (hm : 0 < m) :
    HasSum (fun j : ℕ => 2 * (1 / (2 * ((j + 1 : ℕ) : ℝ) + 1)) *
        ((m - y) / (m + y)) ^ (2 * (j + 1) + 1))
      (gdev y m - (m - y) * ((m - y) / (m + y)) / m) := by
  set v := (m - y) / (m + y) with hv
  have hs : 0 < m + y := by linarith
  have hvabs : |v| < 1 := by
    rw [hv, abs_div, abs_of_pos hs, div_lt_one hs, abs_lt]
    constructor <;> linarith
  have hsum := Real.hasSum_log_sub_log_of_abs_lt_one hvabs
  rw [← hasSum_nat_add_iff' 1] at hsum
  have h1 : 1 + v = 2 * m / (m + y) := by rw [hv]; field_simp; ring
  have h2 : 1 - v = 2 * y / (m + y) := by rw [hv]; field_simp; ring
  have hlog : Real.log (1 + v) - Real.log (1 - v) = -Real.log (y / m) := by
    rw [h1, h2, ← Real.log_div (by positivity) (by positivity), ← Real.log_inv]
    congr 1
    field_simp
  have hval : gdev y m - (m - y) * v / m =
      Real.log (1 + v) - Real.log (1 - v) -
        ∑ i ∈ Finset.range 1, 2 * (1 / (2 * (i : ℝ) + 1)) * v ^ (2 * i + 1) := by
    rw [hlog, Finset.sum_range_one]
    simp only [gdev, Nat.cast_zero, mul_zero, zero_add, pow_one, div_one, mul_one]
    rw [hv]
    have hm' := hm.ne'
    have hs' := hs.ne'
    field_simp
    ring
  rw [hval]
  exact hsum

/-- The direct branch of `gdev` (`y/m` outside `(1/2, 2)`): `|y/m - 1| <= 3.3 gdev(y, m)`, the
claim in the module docstring that makes the direct form relative-accurate. -/
theorem gdev_direct_bound {y m : ℝ} (hy : 0 < y) (hm : 0 < m) (hout : y / m ≤ 1 / 2 ∨ 2 ≤ y / m) :
    |y / m - 1| ≤ 33 / 10 * gdev y m := by
  set r := y / m with hr
  have hr0 : 0 < r := div_pos hy hm
  have hl2a := Real.log_two_gt_d9
  have hl2b := Real.log_two_lt_d9
  unfold gdev
  rw [← hr]
  rcases hout with h | h
  · rw [abs_of_neg (by linarith)]
    have h1 := Real.log_le_sub_one_of_pos (by positivity : 0 < 2 * r)
    rw [Real.log_mul (by norm_num) hr0.ne'] at h1
    nlinarith
  · rw [abs_of_nonneg (by linarith)]
    have h1 := Real.log_le_sub_one_of_pos (by positivity : 0 < r / 2)
    rw [Real.log_div hr0.ne' (by norm_num)] at h1
    nlinarith

/-! ## Gamma, digamma and the residual functions -/

/-- `psi = Gamma'/Gamma`, the digamma function on the positive reals. -/
noncomputable def psi (a : ℝ) : ℝ := deriv Real.Gamma a / Real.Gamma a

/-- `gnorm(a) = a log a - a - log Gamma(a)`. -/
noncomputable def gnorm (a : ℝ) : ℝ := a * Real.log a - a - Real.log (Real.Gamma a)

/-- `psires(n, a) = psi^(n)(a) - d^n log(a)/da^n`. -/
noncomputable def psires (n : ℕ) (a : ℝ) : ℝ := iteratedDeriv n psi a - iteratedDeriv n Real.log a

/-- `sres(n, a) = a^(n+1) psires(n, a)`. -/
noncomputable def sres (n : ℕ) (a : ℝ) : ℝ := a ^ (n + 1) * psires n a

lemma Gamma_eq_re : Real.Gamma = fun x : ℝ => (Complex.Gamma (x : ℂ)).re := by
  funext x
  rw [Complex.Gamma_ofReal, Complex.ofReal_re]

lemma complex_Gamma_differentiableOn :
    DifferentiableOn ℂ Complex.Gamma {s : ℂ | 0 < s.re} := by
  intro s hs
  refine (Complex.differentiableAt_Gamma s fun m h => ?_).differentiableWithinAt
  have : s.re = -(m : ℝ) := by rw [h]; simp
  have hs' : (0 : ℝ) < s.re := hs
  have : (0 : ℝ) ≤ m := Nat.cast_nonneg m
  linarith

lemma complex_Gamma_analyticAt {a : ℝ} (ha : 0 < a) : AnalyticAt ℂ Complex.Gamma (a : ℂ) := by
  apply complex_Gamma_differentiableOn.analyticAt
  apply (isOpen_lt continuous_const Complex.continuous_re).mem_nhds
  simpa using ha

/-- `Real.Gamma` is real analytic on the positive reals. -/
theorem Gamma_analyticAt {a : ℝ} (ha : 0 < a) : AnalyticAt ℝ Real.Gamma a := by
  rw [Gamma_eq_re]
  exact (complex_Gamma_analyticAt ha).re_ofReal

/-- `psi` is real analytic on the positive reals. -/
theorem psi_analyticOnNhd : AnalyticOnNhd ℝ psi (Ioi 0) := by
  intro a ha
  have hG := Gamma_analyticAt (show 0 < a from ha)
  exact hG.deriv.div hG (Real.Gamma_pos_of_pos ha).ne'

/-- `psi` is Mathlib's `Complex.digamma` on the positive reals. -/
theorem psi_eq_digamma {a : ℝ} (ha : 0 < a) : psi a = (Complex.digamma (a : ℂ)).re := by
  have hd : HasDerivAt Complex.Gamma (deriv Complex.Gamma (a : ℂ)) (a : ℂ) :=
    (complex_Gamma_analyticAt ha).differentiableAt.hasDerivAt
  have hr : HasDerivAt Real.Gamma (deriv Complex.Gamma (a : ℂ)).re a := by
    rw [Gamma_eq_re]
    exact hd.real_of_complex
  rw [psi, hr.deriv, Complex.digamma_def, logDeriv_apply, Complex.Gamma_ofReal,
    Complex.div_ofReal_re]

/-- `d log Gamma(a)/da = psi(a)` for `a > 0`. -/
theorem hasDerivAt_log_Gamma {a : ℝ} (ha : 0 < a) :
    HasDerivAt (fun x => Real.log (Real.Gamma x)) (psi a) a :=
  (Gamma_analyticAt ha).differentiableAt.hasDerivAt.log (Real.Gamma_pos_of_pos ha).ne'

/-- `gnorm' = -psires(0, .)` for `a > 0`. -/
theorem hasDerivAt_gnorm {a : ℝ} (ha : 0 < a) : HasDerivAt gnorm (-psires 0 a) a := by
  have h : HasDerivAt gnorm (Real.log a + 1 - 1 - psi a) a :=
    ((Real.hasDerivAt_mul_log ha.ne').sub (hasDerivAt_id' a)).sub (hasDerivAt_log_Gamma ha)
  convert h using 1
  rw [psires, iteratedDeriv_zero, iteratedDeriv_zero]
  ring

/-- `gnorm' = -sres(0, a)/a` for `a > 0` (the generator's `gnorm.fdiff`). -/
theorem hasDerivAt_gnorm_sres {a : ℝ} (ha : 0 < a) : HasDerivAt gnorm (-sres 0 a / a) a := by
  convert hasDerivAt_gnorm ha using 1
  rw [sres, zero_add, pow_one]
  have ha' := ha.ne'
  field_simp

lemma differentiableAt_iteratedDeriv_psi (n : ℕ) {a : ℝ} (ha : 0 < a) :
    DifferentiableAt ℝ (iteratedDeriv n psi) a := by
  rw [iteratedDeriv_eq_iterate]
  exact (psi_analyticOnNhd.iterated_deriv n a ha).differentiableAt

lemma differentiableAt_iteratedDeriv_log (n : ℕ) {a : ℝ} (ha : 0 < a) :
    DifferentiableAt ℝ (iteratedDeriv n Real.log) a := by
  rw [iteratedDeriv_eq_iterate]
  exact (analyticOnNhd_log.iterated_deriv n a ha).differentiableAt

/-- `psires(n)' = psires(n + 1)` for `a > 0` (the generator's `psires.fdiff`). -/
theorem hasDerivAt_psires (n : ℕ) {a : ℝ} (ha : 0 < a) :
    HasDerivAt (psires n) (psires (n + 1) a) a := by
  have h := (differentiableAt_iteratedDeriv_psi n ha).hasDerivAt.sub
    (differentiableAt_iteratedDeriv_log n ha).hasDerivAt
  unfold psires
  rw [iteratedDeriv_succ, iteratedDeriv_succ]
  exact h

/-- `sres(n)' = ((n + 1) sres(n) + sres(n + 1))/a` for `a > 0` (the generator's `sres.fdiff`). -/
theorem hasDerivAt_sres (n : ℕ) {a : ℝ} (ha : 0 < a) :
    HasDerivAt (sres n) (((n + 1) * sres n a + sres (n + 1) a) / a) a := by
  have h : HasDerivAt (sres n)
      ((((n + 1 : ℕ) : ℝ) * a ^ (n + 1 - 1)) * psires n a + a ^ (n + 1) * psires (n + 1) a) a :=
    (hasDerivAt_pow (n + 1) a).mul (hasDerivAt_psires n ha)
  convert h using 1
  rw [sres, sres, Nat.add_sub_cancel]
  push_cast
  have ha' := ha.ne'
  field_simp
  ring

/-- `d^(n+1) log(a)/da^(n+1) = (-1)^n n! a^(-1-n)` for `a > 0`: the terms `psires` subtracts
(`psires_0123`: `p1 - 1/a`, `p2 + 1/a^2`, `p3 - 2/a^3`; the generator's mpmath reference
`(-1)^(n-1) (n-1)!/a^n`). -/
theorem iteratedDeriv_log_succ (n : ℕ) {a : ℝ} (ha : 0 < a) :
    iteratedDeriv (n + 1) Real.log a = (-1) ^ n * n.factorial * a ^ (-1 - n : ℤ) := by
  rw [iteratedDeriv_succ', iteratedDeriv_eq_iterate]
  have hev : deriv Real.log =ᶠ[𝓝 a] Inv.inv := by
    filter_upwards [isOpen_Ioi.mem_nhds ha] with x hx
    exact Real.deriv_log x
  have h2 : deriv^[n] (deriv Real.log) a = deriv^[n] Inv.inv a := by
    rw [← iteratedDeriv_eq_iterate, ← iteratedDeriv_eq_iterate]
    exact (hev.iteratedDeriv n).eq_of_nhds
  rw [h2, iter_deriv_inv]

theorem psires_0123_log_terms {a : ℝ} (ha : 0 < a) :
    iteratedDeriv 1 Real.log a = 1 / a ∧ iteratedDeriv 2 Real.log a = -(1 / a ^ 2) ∧
      iteratedDeriv 3 Real.log a = 2 / a ^ 3 := by
  refine ⟨?_, ?_, ?_⟩
  · rw [iteratedDeriv_log_succ 0 ha]
    norm_num [zpow_neg_one]
  · rw [iteratedDeriv_log_succ 1 ha]
    norm_num [zpow_neg, zpow_ofNat]
  · rw [iteratedDeriv_log_succ 2 ha]
    norm_num [zpow_neg, zpow_ofNat, Nat.factorial]
    ring

/-- The digamma recurrence `psi(x + 1) = psi(x) + 1/x` for `x > 0`. -/
theorem psi_add_one {x : ℝ} (hx : 0 < x) : psi (x + 1) = psi x + x⁻¹ := by
  have hs : ∀ m : ℕ, (x : ℂ) ≠ -m := by
    intro m h
    have := congrArg Complex.re h
    simp at this
    have : (0 : ℝ) ≤ m := Nat.cast_nonneg m
    linarith
  rw [psi_eq_digamma (by linarith), psi_eq_digamma hx, Complex.ofReal_add, Complex.ofReal_one,
    Complex.digamma_apply_add_one _ hs, Complex.add_re, ← Complex.ofReal_inv, Complex.ofReal_re]

/-- The shift recurrence `polygamma_0123` runs, for `n = 0, 1, 2, 3` and every `n`:
`psi^(n)(x + 1) = psi^(n)(x) + (-1)^n n! x^(-1-n)` for `x > 0`. So
`psi(x) = psi(x+1) - 1/x`, `psi'(x) = psi'(x+1) + 1/x^2`, `psi''(x) = psi''(x+1) - 2/x^3` and
`psi'''(x) = psi'''(x+1) + 6/x^4`, the code's `shifted_*` updates. -/
theorem iteratedDeriv_psi_shift (n : ℕ) {x : ℝ} (hx : 0 < x) :
    iteratedDeriv n psi (x + 1) =
      iteratedDeriv n psi x + (-1) ^ n * n.factorial * x ^ (-1 - n : ℤ) := by
  have hev : (fun z => psi (z + 1)) =ᶠ[𝓝 x] (psi + Inv.inv) := by
    filter_upwards [isOpen_Ioi.mem_nhds hx] with z hz
    exact psi_add_one hz
  have h1 : iteratedDeriv n (fun z => psi (z + 1)) x = iteratedDeriv n (psi + Inv.inv) x :=
    (hev.iteratedDeriv n).eq_of_nhds
  have h2 : iteratedDeriv n psi (x + 1) = iteratedDeriv n (psi + Inv.inv) x := by
    rw [← h1, iteratedDeriv_comp_add_const]
  rw [h2, iteratedDeriv_add ((psi_analyticOnNhd x hx).contDiffAt) (contDiffAt_inv ℝ hx.ne'),
    iteratedDeriv_eq_iterate (f := Inv.inv), iter_deriv_inv]

end SuperGLM.Issue544.Primitives

#print axioms SuperGLM.Issue544.Primitives.expit_branch
#print axioms SuperGLM.Issue544.Primitives.softplus_branch
#print axioms SuperGLM.Issue544.Primitives.invexpm1_branch
#print axioms SuperGLM.Issue544.Primitives.hasDerivAt_softplus
#print axioms SuperGLM.Issue544.Primitives.hasDerivAt_expit
#print axioms SuperGLM.Issue544.Primitives.hasDerivAt_log1mexp
#print axioms SuperGLM.Issue544.Primitives.hasDerivAt_invexpm1
#print axioms SuperGLM.Issue544.Primitives.hasDerivAt_gdev_y
#print axioms SuperGLM.Issue544.Primitives.hasDerivAt_gdev_m
#print axioms SuperGLM.Issue544.Primitives.gdev_log_branch
#print axioms SuperGLM.Issue544.Primitives.gdev_window
#print axioms SuperGLM.Issue544.Primitives.gdev_loader_series
#print axioms SuperGLM.Issue544.Primitives.gdev_direct_bound
#print axioms SuperGLM.Issue544.Primitives.Gamma_analyticAt
#print axioms SuperGLM.Issue544.Primitives.psi_analyticOnNhd
#print axioms SuperGLM.Issue544.Primitives.psi_eq_digamma
#print axioms SuperGLM.Issue544.Primitives.hasDerivAt_log_Gamma
#print axioms SuperGLM.Issue544.Primitives.hasDerivAt_gnorm
#print axioms SuperGLM.Issue544.Primitives.hasDerivAt_gnorm_sres
#print axioms SuperGLM.Issue544.Primitives.hasDerivAt_psires
#print axioms SuperGLM.Issue544.Primitives.hasDerivAt_sres
#print axioms SuperGLM.Issue544.Primitives.iteratedDeriv_log_succ
#print axioms SuperGLM.Issue544.Primitives.psires_0123_log_terms
#print axioms SuperGLM.Issue544.Primitives.psi_add_one
#print axioms SuperGLM.Issue544.Primitives.iteratedDeriv_psi_shift
