# Shape constraints, caps, constrained interactions, adaptive smoothing and LSS constraints: literature sweep

Four read-only sweeps run on 2026-11-02, one per candidate project. Citations
were located in that session; "UNVERIFIED" marks claims the sweep could not
confirm, "derived" marks our own algebra. No GPL source was opened (scam,
mgcv, cgam, gamlss, bamlss, mboost, brms, cirls are GPL; judged from papers
and documentation pages only). cpsplines is MIT, pyGAM and OSQP Apache-2.0.

Notation: k coefficients, degree d, D̃_j the knot-scaled j-th difference
matrix. Lebair & Shen (2015, arXiv 1510.05314, Lemma 2.1): f^(j) is a degree
d−j spline with coefficients D̃_j β, and a degree-0 or degree-1 spline is
nonnegative iff its coefficients are. So rows D̃_j β ≥ 0 are exact when
d − j ≤ 1 and only sufficient otherwise.

## Cross-cutting finding

No sweep found a published REML/LAML method for smoothing selection with
active inequality constraints, in GLMs, tensors or GAMLSS. scam uses
GCV/UBRE "rather than REML" (Pya Arnqvist 2024, arXiv 2403.09438, §3.4);
mgcv's `scasm` (Wood 2026, Stat. Modelling 26(4)) uses LP feasible start,
sequential QP and EFS. The established workaround is to treat the active set
as equalities (Wood 1994, SIAM J. Sci. Comput. 15(5), via secondary sources;
Wood on r-help 2007, who warns convergence "can be tricky"). Derived: with Z
spanning the null space of the active rows, the KKT point is stationary in
the reduced coordinates, so EFS holds with ZᵀS_jZ and ZᵀHZ for a fixed face;
the score is piecewise smooth in ρ with kinks where the face changes, so
freeze or add hysteresis to the face between outer steps and disclose
non-convergence. superglm today uses a weaker two-stage heuristic
(unconstrained REML, then constrained refit; `model/fit_ops.py`).

## 1. Two shapes on one term, and caps

- Two shapes on a cubic: curvature rows are exact (k−2 rows). With concavity,
  f' is decreasing, so increasing ⇔ f'(right end) ≥ 0: one extra row
  (Wang, Fan, Li & Liu 2023, arXiv 2307.01748, Prop. 1, stated for
  increasing-convex; the other three follow by symmetry). Stacking the full
  k−1 monotone rows gives the same feasible set with k−2 redundant rows,
  which make the working set degenerate for N&W Alg. 16.3: drop them.
- Monotone alone on a cubic is only sufficient; exact conditions are
  nonlinear (Wang et al. Prop. 1). Meyer (2008, AoAS 2(3)) recommends
  quadratic I-splines where exact monotonicity matters.
- Caps: convex-hull rows L ≤ c_j − B(x_base)β ≤ U are sufficient; the gap is
  O(h²‖f''‖) (Lyche & Mørken, *Spline Methods*, Lemma 9.17, Thm 9.18; Cohen &
  Schumaker 1985). Exact and cheap: categorical / ordered levels and
  adjacent steps (2 rows each), caps at exported rating-table band points,
  range caps on a monotone term (2 endpoint rows), upper cap on a convex or
  lower cap on a concave term (endpoints). Slope caps exact on quadratics.
- Exact caps on a whole cubic curve: no linear form. SOS/SDP (Papp &
  Alizadeh 2014, JCGS 23(1); Navarro-García et al. 2023, 2024, needs MOSEK)
  does not fit an active-set QP. Recommended linear routes: knot-insertion
  refinement (rows on Tβ, banded) or lazy rows in the style of Turlach
  (2005): solve, find each interval's cubic extremum analytically, add a
  point row, warm-start. Point rows are necessary, so the converged fit is
  exact to tolerance.
- Feasible start: caps whose interval contains relativity 1 keep β = 0
  feasible; caps on change from the current tariff may need a phase-1 start
  (`_homogeneous_feasible_start` in `solvers/constrained_qp.py` returns None
  for b ≠ 0 today).
- Inference: point estimate equals the active-set-as-equalities fit (Meyer
  2008, Prop. 1). Face-conditional covariance Z(ZᵀHZ)⁻¹Zᵀ under-states
  uncertainty near the boundary; face mixtures (Meyer & Liao 2022; Meyer
  2018, Stat. Sci. 33(4)) or truncated-normal sampling (Geweke 1996) are the
  honest options.
- Actuarial practice: Werner & Modlin, *Basic Ratemaking* (5th ed., 2016),
  ch. 14, caps the change from the current relativity and off-balances the
  base rate iteratively (Tables 14.10–14.13). CAS Monograph 5 (2025 rev.)
  §2.6, §9.1: fixed factors as offsets, rest refit. Wüthrich (2026, arXiv
  2606.07276; not peer-reviewed, ChatGPT-generated proofs disclosed): fit-time
  balance constraint beats intercept-shift post-processing in KL. No paper on
  fit-time relativity caps found.

## 2. Constrained interactions

- Standard rows: monotone in x1 is (D1 ⊗ I_k2)β ≥ 0, (k1−1)k2 rows; both
  directions 2k1k2 − k1 − k2 rows (Bollaerts, Eilers & Van Mechelen 2006,
  BJMSP 59, §2.2; Hofner, Kneib & Hothorn 2016, arXiv 1403.7118, §4.3; scam).
  Derived: exact only for degree (q1 ≤ 2, q2 ≤ 1); cubic margins are
  sufficient only. Exact degree-(1,1) warped planes: cgam (Liao & Meyer 2019,
  JSS 89(5)); multilinear lattices (Gupta et al. 2016, JMLR).
- Conservatism: de Boor & Daniel (1974) O(h²) approximation for nonnegative
  coefficients; Arai & Kitahara (2026, arXiv 2605.03711) 10–30% higher
  objective than exact in 1-D, about 30× faster. Statistically benign (Lebair
  & Shen 2015, uniform consistency). Liao & Meyer's "severely over-constrain"
  critique of scam is scam's construction, not the condition: 4(x1 + x2 −
  x1x2) is feasible under the difference rows (derived, Marsden's identity).
- Total effect across main effect + interaction: Hofner et al. (2016, §4.3)
  and Hu et al. (2023, arXiv 2309.02426) say the full surface must be
  constrained; scam's `tismi`/`tismd` constrain the interaction alone. No
  published treatment of rows on summed coefficients of separately centred
  terms: ours. Derived recipe: with a shared x1 basis, partition of unity
  gives Σij (a_i + b_j + c_ij) B_i C_j; monotone in x1 is
  (D1 a) ⊗ 1 + (D1 ⊗ I) c ≥ 0, still (k1−1)k2 rows, f2 drops out; compose
  through the identifiability maps Z1, Z1 ⊗ Z2; differing knots re-expressed
  on the knot union (nested spaces). Needs a model-level constraint set:
  today `irls_direct/_qp.py` stacks per-group blocks only.
- Factor-by: separate curves L(k−1) block rows; shared main effect plus
  deviations D1(a + d_l) ≥ 0, arrow structure. Ordered categorical × smooth:
  (L−1)k rows, exact only for a linear x margin.
- Speed: at 20×20, 380–760 rows; an add/drop is O(p²) ≈ 1e5 flops against
  O(np²) for X'WX, so outer iteration counts dominate, not the QP. Liao &
  Meyer: cone projection 17 s vs scam 611 s (n = 100). SCAM reparametrisation
  needs 2–4× the O(np²) steps (Pya Arnqvist 2024 §3.4); Wood moved `scasm`
  to linear inequalities. Derived: single-direction rows have full row rank,
  so δ = Aβ turns them into simple bounds and projected Newton applies
  (Bertsekas 1982). Both directions: more rows than rank; Goldfarb–Idnani
  (1983) dual active set from the existing Cholesky factor, or parametric
  active set (qpOASES, Ferreau et al. 2014). OSQP (Stellato et al. 2020) is
  modest-accuracy plus polishing. Isotonic algorithms need a diagonal
  Hessian; usable only as a projection inside ADMM.
- Inference: face-conditional (Liao & Meyer 2019 §2; Lebair & Shen eq. 11);
  `scasm` offers constraint-respecting bootstrap intervals.
- No permissively licensed hard-constrained tensor-smooth GLM found.

## 3. Adaptive smoothing

- Standard and nearly free for us: linear-in-λ varying penalty
  S = Σ_l λ_l D′ diag(ψ_l) D, ψ_l a B-spline basis over the difference
  positions (Ruppert & Carroll 2000, ANZJS 42; mgcv `bs="ad"`, default m = 5,
  k = 40; Rodríguez-Álvarez, Durbán, Lee & Eilers 2019, Stat. Comput. 29,
  §§4.1, 5.1, 6; Wood & Fasiolo 2017, Biometrics 73(4), §4). Null space is
  independent of λ (B-splines sum to one), so EFS Theorem 1 holds. Derived:
  log|D′WD|₊ = Σ_j log w_j + log det(DD′), closed-form gradient and Hessian.
- Failure modes: unpenalised weights are sensitive to m (SOP §6: large m
  gives local linear fits); weakly identified λ on small data (Wood &
  Fasiolo §4); piecewise-constant ρ over-fits above about 8 pieces (Wang, Du &
  Shen, arXiv 1306.1868, §5). Default m = 5, range 3–8.
- Alternatives that do not fit the estimator: Krivobokova, Crainiceanu &
  Kauermann (2008, JCGS) hierarchical log-variance; trend filtering
  (Tibshirani 2014, AoS 42(1)); BARS; A-splines (arXiv 1808.01770); NEG and
  horseshoe priors.
- Evidence: a single λ already widens where data are thin (Wang, Du & Shen
  equivalent kernel). On freMTPL DriverAge, Seck & Denuit (2021, LIDAM DP
  2021/35) found adaptive and plain P-spline age effects "almost identical"
  (in-sample). No held-out insurance comparison found: gate any adoption on
  held-out deviance on the real book.

## 4. Shape constraints in SuperLSS

- Every built-in LSS family is mean-parametrised, so a constraint on the
  mean predictor constrains the price exactly.
- Stochastic order: Delong, Gatti & Wüthrich (arXiv 2503.18896, Prop. 3.4)
  for EDMs with φ and p fixed in that factor: μ1 ≤ μ2 gives likelihood-ratio
  and hence usual stochastic order, so layer costs, quantiles and TVaR are
  monotone too. Derived: fixed-shape scale families (gamma, log-normal,
  generalized gamma) have Q_τ = μ q_τ(shape), same conclusion. Location-scale
  quantiles with unbounded error support force σ′ = 0. Practical rule: a
  factor that must be stochastically monotone stays out of the scale and
  shape predictors.
- Method: linearly constrained SQP / active-set Newton on the joint problem,
  rows on the mean block only; because constraints are linear the Lagrangian
  Hessian is the objective Hessian, so each step is a plain QP (CIRLS,
  Masselot et al. 2025, arXiv 2509.18406, §2.2, single-predictor GLM).
  Backtracking stays feasible (convex set). Null-space Z = blockdiag(Z_μ, I).
  Outside assumptions: joint LSS likelihood is not concave; the reduced
  Hessian ZᵀHZ is what must be positive definite.
- Prior art: boosting (Hofner et al. §3.3, QP base learners in gamboostLSS);
  gamlss `pbm()` (asymmetric penalty, mechanism UNVERIFIED); bamlss `"ms"`
  (boosting and backfitting only); Jiao et al. (arXiv 2503.00254) say GAMLSS
  has "no direct way yet to ensure shape restrictions". No joint
  penalised-Newton GAMLSS with shape constraints and REML/EFS found: ours.
- Inference: CIRLS truncated MVN with under-coverage near constraints;
  default face-conditional covariance with disclosure, TMVN behind an option.
