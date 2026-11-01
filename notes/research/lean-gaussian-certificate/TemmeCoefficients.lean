import Mathlib

/-!
# Issue 544, stage 2: the coefficients `d_n` of `eta/(lambda - 1)`

`src/superglm/distributional/kernels/_gg_uniform_tail.py` stores `_D`, 37 coefficients
`d_0 .. d_36` of the Taylor series of `eta/(lambda - 1)` in Temme's variable `eta`
(`eta^2/2 = lambda - 1 - log lambda`, DLMF 8.12), and `_G`, 29 coefficients of
`G(t) = sqrt(2 (e^t - 1 - t)/t^2)`, each an exact rational rounded once to float64.

With `t = log lambda`, `eta = t G(t)` and `eta/(lambda - 1) = t G(t)/(e^t - 1)`
(`GGChangeOfVariables.lean`). So the series `D(eta) = sum d_n eta^n` is characterised by
two identities between power series in `t`:

* `G(t)^2 = g(t) = sum_k 2 t^k/(k + 2)!` with `G(0) = 1 > 0`;
* `D(t G(t)) · (e^t - 1)/t = G(t)`.

This file checks both identities through `t^36` in exact rational arithmetic, for the
tables `GTab` (37 entries) and `dTab` (37 entries) below, by kernel evaluation
(`decide +kernel`, no `native_decide`). The truncated product `mulT` is shown to be the
Cauchy product (`coef_mulT`), and composition is Horner's rule in `ℚ[t]/(t^37)`.

Outside Lean, a short script (recorded in
`notes/research/2026-11-01-issue-544-proofs.md`) checked that Python's correctly rounded
`float()` of every `dTab` entry equals the engine's `_D` literal (37 of 37), and of the
first 29 `GTab` entries the engine's `_G` (29 of 29).

What is not proved here: that a formal power-series identity determines the Taylor
series of the real-analytic function (the standard composition theorem for convergent
series), and uniqueness beyond order 2 (both identities are triangular in the unknown
coefficients, so they determine them, but only orders 0-2 are proved to be forced, in
`low_orders_forced`).

Repository base: ff2ed52d (branch `claude/issue-544-gamma-bounds`). Date: 2026-11-01.
-/

namespace SuperGLM.Issue544.Temme

open Finset

/-- Coefficient `k` of a truncated series stored as a list (zero past its end). -/
def coef (a : List ℚ) (k : ℕ) : ℚ := a.getD k 0

/-- Coefficientwise sum. -/
def addL : List ℚ → List ℚ → List ℚ
  | [], b => b
  | a, [] => a
  | x :: xs, y :: ys => (x + y) :: addL xs ys

/-- `mulT n a b`: the product `a b` truncated after `t^(n-1)`. -/
def mulT : ℕ → List ℚ → List ℚ → List ℚ
  | 0, _, _ => []
  | _ + 1, [], _ => []
  | n + 1, x :: xs, b => addL ((b.map (x * ·)).take (n + 1)) (0 :: mulT n xs b)

/-- `compT n d e = d(e(t))` truncated after `t^(n-1)`, by Horner's rule. -/
def compT (n : ℕ) (d e : List ℚ) : List ℚ :=
  d.foldr (fun c acc => addL [c] (mulT n e acc)) []

/-- `a` and `b` agree in coefficients `0 .. n-1`. -/
def agreeUpTo (n : ℕ) (a b : List ℚ) : Bool :=
  (List.range n).all fun k => coef a k == coef b k

/-- `g(t) = 2 (e^t - 1 - t)/t^2 = sum_k 2 t^k/(k + 2)!`, first `n` coefficients. -/
def gSeries (n : ℕ) : List ℚ := (List.range n).map fun k => 2 / (Nat.factorial (k + 2) : ℚ)

/-- `(e^t - 1)/t = sum_k t^k/(k + 1)!`, first `n` coefficients. -/
def expm1OverT (n : ℕ) : List ℚ := (List.range n).map fun k => 1 / (Nat.factorial (k + 1) : ℚ)

/-! ## Semantics of the list operations -/

lemma coef_nil (k : ℕ) : coef [] k = 0 := by simp [coef]

lemma coef_cons_zero (x : ℚ) (xs : List ℚ) : coef (x :: xs) 0 = x := by simp [coef]

lemma coef_cons_succ (x : ℚ) (xs : List ℚ) (k : ℕ) : coef (x :: xs) (k + 1) = coef xs k := by
  simp [coef]

lemma coef_addL : ∀ (a b : List ℚ) (k : ℕ), coef (addL a b) k = coef a k + coef b k
  | [], b, k => by simp [addL, coef_nil]
  | x :: xs, [], k => by simp [addL, coef_nil]
  | x :: xs, y :: ys, 0 => by simp [addL, coef_cons_zero]
  | x :: xs, y :: ys, k + 1 => by simp [addL, coef_cons_succ, coef_addL xs ys k]

lemma coef_map_mul (c : ℚ) : ∀ (b : List ℚ) (k : ℕ), coef (b.map (c * ·)) k = c * coef b k
  | [], k => by simp [coef_nil]
  | y :: ys, 0 => by simp [coef_cons_zero]
  | y :: ys, k + 1 => by simp [coef_cons_succ, coef_map_mul c ys k]

lemma coef_take : ∀ (l : List ℚ) (m k : ℕ), k < m → coef (l.take m) k = coef l k
  | [], m, k, _ => by simp [coef_nil]
  | y :: ys, 0, k, h => absurd h (Nat.not_lt_zero k)
  | y :: ys, m + 1, 0, _ => by simp [coef_cons_zero]
  | y :: ys, m + 1, k + 1, h => by
      simp only [List.take_succ_cons, coef_cons_succ]
      exact coef_take ys m k (by omega)

/-- `mulT` is the Cauchy product below the truncation order. -/
theorem coef_mulT : ∀ (n : ℕ) (a b : List ℚ) (k : ℕ), k < n →
    coef (mulT n a b) k = ∑ i ∈ range (k + 1), coef a i * coef b (k - i)
  | 0, a, b, k, h => absurd h (Nat.not_lt_zero k)
  | n + 1, [], b, k, _ => by simp [mulT, coef_nil]
  | n + 1, x :: xs, b, 0, h => by
      simp [mulT, coef_addL, coef_take _ _ _ h, coef_map_mul, coef_cons_zero]
  | n + 1, x :: xs, b, k + 1, h => by
      rw [mulT, coef_addL, coef_take _ _ _ h, coef_map_mul, coef_cons_succ,
        coef_mulT n xs b k (by omega),
        sum_range_succ' (fun i => coef (x :: xs) i * coef b (k + 1 - i))]
      simp only [coef_cons_succ, coef_cons_zero, Nat.add_sub_add_right, Nat.sub_zero]
      ring

/-- `compT` is Horner's rule: `d(e) = d_0 + e · (d_1 + e · (...))`. -/
theorem compT_cons (n : ℕ) (c : ℚ) (d e : List ℚ) :
    compT n (c :: d) e = addL [c] (mulT n e (compT n d e)) := rfl

lemma agreeUpTo_spec {n : ℕ} {a b : List ℚ} (h : agreeUpTo n a b = true) :
    ∀ k < n, coef a k = coef b k := by
  intro k hk
  simp only [agreeUpTo, List.all_eq_true, List.mem_range, beq_iff_eq] at h
  exact h k hk

/-! ## The tables -/

/-- `G_0 .. G_36`, the Taylor coefficients of `G = sqrt g`; the engine's `_G` is the first 29,
rounded. -/
def GTab : List ℚ := [
    (1 : ℚ),
    (1 : ℚ) / 6,
    (1 : ℚ) / 36,
    (1 : ℚ) / 270,
    (1 : ℚ) / 2592,
    (17 : ℚ) / 544320,
    (11 : ℚ) / 5443200,
    (1 : ℚ) / 8164800,
    (1 : ℚ) / 111974400,
    (487 : ℚ) / 775982592000,
    (457 : ℚ) / 32591268864000,
    (-1667 : ℚ) / 635529742848000,
    (-3617 : ℚ) / 30505427656704000,
    (1349 : ℚ) / 36606513188044800,
    (1733 : ℚ) / 422382844477440000,
    (-36371 : ℚ) / 148505968274227200000,
    (-7408369 : ℚ) / 104548201665055948800000,
    (-10699531 : ℚ) / 11918494989816378163200000,
    (11102667353 : ℚ) / 11799310039918214381568000000,
    (251866777 : ℚ) / 3539793011975464314470400000,
    (-53479057901 : ℚ) / 5946852260118780048310272000000,
    (-6524547501877 : ℚ) / 4103328059481958233334087680000000,
    (9459902453891 : ℚ) / 320059588639592742200058839040000000,
    (1093399212883 : ℚ) / 43644489359944464845462568960000000,
    (29099494301 : ℚ) / 24883155579365810861034700800000000,
    (-2763963905041 : ℚ) / 9405832809000276505471116902400000000,
    (-100262673734953 : ℚ) / 2765314845846081292608508369305600000000,
    (112704677582753 : ℚ) / 55760400458497019134668755215319040000000,
    (2690366085955648393 : ℚ) / 4001366336901746093103829874251294310400000000,
    (50251788592309509773 : ℚ) / 3721270693318623866586561783053703708672000000000,
    (-17511876960322871629163 : ℚ) / 1897848053592498171959146509357388891422720000000000,
    (-328501437999887900411 : ℚ) / 406681725769821036848388537719440476733440000000000,
    (1932650175232567309153 : ℚ) / 22309397527944468307111599783466449009377280000000000,
    (364536524482003998898429 : ℚ) / 20301551750429466159471555802954468598533324800000000000,
    (-1888285922910947218429 : ℚ) / 13957472994202953595914844395729681426634726440960000000,
    (-429823133016931383128786363 : ℚ) / 1519228822353922783107067276306740432009023402803200000000000,
    (-5806388400769506267424217119247 : ℚ) / 348936475918248984824031212022132142423832495155838976000000000000]

/-- `d_0 .. d_36`, the Taylor coefficients of `eta/(lambda - 1)` in `eta`; the engine's `_D`,
rounded. -/
def dTab : List ℚ := [
    (1 : ℚ),
    (-1 : ℚ) / 3,
    (1 : ℚ) / 12,
    (-2 : ℚ) / 135,
    (1 : ℚ) / 864,
    (1 : ℚ) / 2835,
    (-139 : ℚ) / 777600,
    (1 : ℚ) / 25515,
    (-571 : ℚ) / 261273600,
    (-281 : ℚ) / 151559100,
    (163879 : ℚ) / 197522841600,
    (-5221 : ℚ) / 29554024500,
    (5246819 : ℚ) / 782190452736000,
    (5459 : ℚ) / 531972441000,
    (-534703531 : ℚ) / 122021710626816000,
    (91207079 : ℚ) / 99704934754425000,
    (-4483131259 : ℚ) / 175711263302615040000,
    (-2650986803 : ℚ) / 45465450248017800000,
    (432261921612371 : ℚ) / 17743323368298066739200000,
    (-6171801683 : ℚ) / 1227567156696480600000,
    (6232523202521089 : ℚ) / 56636688191607429031526400000,
    (4283933145517 : ℚ) / 12705320071808574210000000,
    (-25834629665134204969 : ℚ) / 185541790515705937507280486400000,
    (11963983648109 : ℚ) / 419275562369682948930000000,
    (-1579029138854919086429 : ℚ) / 3072572050940090325120564854784000000,
    (-208697624924077 : ℚ) / 105657441717160103130360000000,
    (746590869962651602203151 : ℚ) / 921771615282027097536169456435200000000,
    (-29320119130515566117 : ℚ) / 177455371374430493811049182600000000,
    (1511513601028097903631961 : ℚ) / 597308006702753559203437807770009600000000,
    (2700231121460756431181 : ℚ) / 231046893529508502941986035745200000000,
    (-8849272268392873147705987190261 : ℚ) / 1855178938018082279529957487152872816640000000000,
    (10084288256532215186381 : ℚ) / 10397110208827882632389371608534000000000,
    (-6208770108287283939483943525987 : ℚ) / 480088045177548944685317694066691261071360000000000,
    (-6782242429223267933535073 : ℚ) / 97316951554628981439164518255878240000000000,
    (2355444393109967510921431436000087153 : ℚ) / 83080196394065199975683597593629056110920990720000000000,
    (-51748587106835353426330148693 : ℚ) / 8998217291595659510809468851493269705120000000000,
    (2346608607351903737647919577082115121863 : ℚ) / 34544745660652310149889239879430961530920947941376000000000000]

theorem GTab_length : GTab.length = 37 := by decide +kernel

theorem dTab_length : dTab.length = 37 := by decide +kernel

/-- `G(0) = 1`, the positive branch of the square root. -/
theorem GTab_zero : coef GTab 0 = 1 := by decide +kernel

/-- `d_2 = +1/12` (Gil, Segura and Temme's arXiv v1, eq. 2.37, prints `-1/12`). -/
theorem dTab_two : coef dTab 2 = 1 / 12 := by decide +kernel

theorem dTab_low : coef dTab 0 = 1 ∧ coef dTab 1 = -1 / 3 ∧ coef dTab 2 = 1 / 12 ∧
    coef dTab 3 = -2 / 135 ∧ coef dTab 4 = 1 / 864 := by decide +kernel

lemma G_sq_check : agreeUpTo 37 (mulT 37 GTab GTab) (gSeries 37) = true := by decide +kernel

lemma d_comp_check :
    agreeUpTo 37 (mulT 37 (compT 37 dTab (0 :: GTab)) (expm1OverT 37)) GTab = true := by
  decide +kernel

/-- `G^2 = g` through `t^36`: `sum_{i <= k} G_i G_{k-i} = 2/(k + 2)!` for every `k <= 36`. -/
theorem G_sq_eq_g : ∀ k < 37,
    ∑ i ∈ range (k + 1), coef GTab i * coef GTab (k - i) = 2 / (Nat.factorial (k + 2) : ℚ) := by
  intro k hk
  rw [← coef_mulT 37 GTab GTab k hk, agreeUpTo_spec G_sq_check k hk]
  simp [coef, gSeries, hk]

/-- `D(t G(t)) · (e^t - 1)/t = G(t)` through `t^36`: with `eta = t G(t)` and
`lambda - 1 = e^t - 1`, this is `D(eta) = eta/(lambda - 1)` as power series in `t`. -/
theorem d_certificate : ∀ k < 37,
    coef (mulT 37 (compT 37 dTab (0 :: GTab)) (expm1OverT 37)) k = coef GTab k :=
  agreeUpTo_spec d_comp_check

/-- Orders 0-2 are forced: any `G_0 > 0`, `G_1`, `G_2`, `d_0`, `d_1`, `d_2` satisfying the two
identities through `t^2` have `G = (1, 1/6, 1/36)` and `d = (1, -1/3, 1/12)`. So `d_2 = -1/12`
satisfies neither. -/
theorem low_orders_forced (G0 G1 G2 d0 d1 d2 : ℚ) (hpos : 0 < G0)
    (hG : agreeUpTo 3 (mulT 3 [G0, G1, G2] [G0, G1, G2]) (gSeries 3) = true)
    (hd : agreeUpTo 3 (mulT 3 (compT 3 [d0, d1, d2] [0, G0, G1, G2]) (expm1OverT 3))
      [G0, G1, G2] = true) :
    (G0 = 1 ∧ G1 = 1 / 6 ∧ G2 = 1 / 36) ∧ (d0 = 1 ∧ d1 = -1 / 3 ∧ d2 = 1 / 12) := by
  have hg : gSeries 3 = [1, 1 / 3, 1 / 12] := by decide +kernel
  have hu : expm1OverT 3 = [1, 1 / 2, 1 / 6] := by decide +kernel
  rw [hg] at hG
  rw [hu] at hd
  have g0 := agreeUpTo_spec hG 0 (by norm_num)
  have g1 := agreeUpTo_spec hG 1 (by norm_num)
  have g2 := agreeUpTo_spec hG 2 (by norm_num)
  have e0 := agreeUpTo_spec hd 0 (by norm_num)
  have e1 := agreeUpTo_spec hd 1 (by norm_num)
  have e2 := agreeUpTo_spec hd 2 (by norm_num)
  simp [mulT, compT, addL, coef] at g0 g1 g2 e0 e1 e2
  have hG0 : G0 = 1 := by nlinarith
  subst hG0
  have hG1 : G1 = 1 / 6 := by linarith
  subst hG1
  have hG2 : G2 = 1 / 36 := by linarith
  subst hG2
  refine ⟨⟨rfl, rfl, rfl⟩, ?_⟩
  have hd0 : d0 = 1 := by linarith
  subst hd0
  have hd1 : d1 = -1 / 3 := by linarith
  subst hd1
  refine ⟨rfl, rfl, ?_⟩
  linarith

end SuperGLM.Issue544.Temme

#print axioms SuperGLM.Issue544.Temme.coef_mulT
#print axioms SuperGLM.Issue544.Temme.GTab_length
#print axioms SuperGLM.Issue544.Temme.dTab_length
#print axioms SuperGLM.Issue544.Temme.GTab_zero
#print axioms SuperGLM.Issue544.Temme.dTab_two
#print axioms SuperGLM.Issue544.Temme.dTab_low
#print axioms SuperGLM.Issue544.Temme.G_sq_eq_g
#print axioms SuperGLM.Issue544.Temme.d_certificate
#print axioms SuperGLM.Issue544.Temme.low_orders_forced
