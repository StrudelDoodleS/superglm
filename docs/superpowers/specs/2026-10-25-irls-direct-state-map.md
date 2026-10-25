# `_fit_irls_direct_once`: state map (S0)

Date: 2026-10-25. Branch `claude/irls-direct-split` at 6edff3e6 (master 5d295301 plus the
solver-entry refusal). File `src/superglm/solvers/irls_direct.py`, function lines
**1172–5170** (3,999 lines). Everything here was re-derived at 5d295301, after #527 and #530.
These line numbers supersede the ones in the anatomy (`audit042/irls_split/anatomy/report.md`,
at f76a7265) and in the design (`2026-10-23-irls-direct-split-design.md`).

How each fact was found:
- **[AST]**: an `ast` + `symtable` pass over the function. The scratch scripts are in
  `audit042/irls_split/s0_out/analysis/` and are not committed.
- **[blame]**: `git blame -w -M` on the branch, with commits mapped to PR numbers.
- **[read]**: I read the code.
- **[run]**: the S0 characterisation harness (`benchmarks/irls_direct_characterisation.py`).

## 0. One screen

- **Shape.** [AST]
  - 47 parameters (the anatomy had 48; #527 removed `_scop_joint`) and 360 locals (410).
  - 26 nested `def`s (993 lines; the largest is `true_mode_residual`, 354 lines), 14 lambdas
    and 18 generator expressions.
  - 257 `if`s, nesting depth 8.
  - 212 names cross a block boundary. Apart from a few reused temporaries (`g`, `gi`, `st`, `centered`, `z_off`, `group`, `certification`, `certified`),
    those are the fit object's fields.
  - Setup is 1172–2966 (1,795 lines), the loop 2967–4387 (1,421) and finalisation 4388–5170
    (783).
- **Blame.** [blame] #156 669 lines, **#437 574**, #425 428, #147 415, #165 214, **#527 173**,
  #407 120, #498 119, #439 93. Commits without a PR (38bf9706, 0987cb7f) 181.
- **#527 changed SCOP** in five ways: a joint Newton step, observed-Newton rows, a latent maximum
  step, damped trials that re-solve the ordinary block, and terminal curvature without a step.
  The design's "SCOP block-coordinate" mode and `_scop_joint` no longer exist. SCOP is now a
  client of the curvature policy and of the observed-row predicate (§1).
- **Interface (§6).**
  - One step strategy per fit: `QRStep`, `SCOPStep`, `StructuredStep` or `DenseStep`, with the
    QP inside `DenseStep`.
  - Five collaborators: `PenaltyOps`, `StateEvaluator`, `CenteredSystemBuilder`,
    `CurvaturePolicy` and `MeanSpaceOverlay`.
  - A shared `TrialSearch` and a `FeasibilityContract`, a `Convergence` unit, a `Recorder`, and
    two finalisers (centred and structured).
- **The hazards S1 must not break (§5):**
  - **Late binding:** `_score_centre`, `_observed_newton_active`, `_can_reuse_weighted_gram`,
    `_state_center` and `mode_bar`.
  - **Bound methods:** `retained_rank_info` and `certify_centered_factor`, which outlive the
    solve.
  - **Identity keys and checks:** `scop_merit_errors` (`id()`), the `scop_trial_cache` float keys,
    `SCOPRunCentring.bind` (by identity), `_initial_data_key` (`id()`), the identity checks in
    the factor-certificate cache, and F04's `_judged_eta is eta_unclipped`.
- **Length (§7).** The function is not lines-of-code abuse: 19% is comments and docstrings, and
  copy-paste is negligible. It is long because four step methods and one overlay are
  multiplexed by `if` through one scope.
  - SCOP has 685 exclusive lines and the Binomial/log mean-space machinery 577.
  - Instrumentation is 494, the structured route 288, the centred-system caches 236,
    convergence 220, dense 206, the QP 127 and QR 47.
  - The rest is shared.

## 1. What changed since the anatomy (#527, #530, and #525/#526 before them)

[read] Everything below is in the current code.

- **SCOP step (L04, 3088–3220).** It solves the ordinary (non-SCOP) block on the reduced design
  with the latent coordinates held. `build_centered_system` builds that system directly, not
  through `get_centered_system`. When the Gram is not authoritative a factor certificate decides.
  `SCOPJointNewton.step` then couples the two blocks through the Schur complement (Pya & Wood
  2015). The block Gauss–Seidel step and its theta-only search are deleted.
- **Damped SCOP trials (L09).**
  - A damped trial re-solves the ordinary block at its own latent coordinates
    (`_SCOPOrdinarySolve`, built in L04, captured by `evaluate_scop_trial`).
  - A latent step is first shortened to `_LATENT_MAX_STEP` (`_latent_step_depth`), and the
    decision is relabelled in fractions of the full step.
  - Trial predictors are centred and carry `(level, centre)` (`_SCOPTrialState.level/centre`,
    `_level_about`).
- **SCOP curvature.** `_observed_newton_active` (S15, 2213) no longer excludes `_has_scop`, so
  SCOP takes observed rows wherever the pair supports them. `_stagnation_newton_available` and
  `_can_reuse_weighted_gram` still exclude SCOP. The SCOP search passes
  `invalid_state=observed_rows_invalid` (S15b, 2238, now a fit-level closure shared with
  L11's default). The SCOP search still has no mean-space or link-domain guard (anatomy
  surprise 1, now partly closed).
- **SCOP stop (L16).** A damped SCOP step (`n_halvings > 0`) never stops the fit, under the
  coefficient rule or the objective rule.
- **SCOP merit (S22).** NB2's unit-deviance rounding (`nb2_unit_roundoff`) enters the forward-error
  allowance, and `merit_gamma` comes from `superglm._roundoff` (#530).
- **SCOP terminal curvature (F05, 4551–4581).** `SCOPJointNewton.block_curvature` replaces the
  discarded Newton step (anatomy surprise 16 is retired).
- **SCOP latent certificate (S08, 1908–1955).** The joint step hands every group its columns of
  one set of discarded directions. The certificate reassembles them and projects once, and
  refuses mismatched counts as an invariant.
- **Run-shared SCOP state.** `SCOPRunCentring` now also carries each lone group's exact support
  (`_lone_scop_support`). It is bound before the per-group state is built (S16, 2257–2271).
- **Docstring drift (new).** The `_use_observed_newton` docstring (1325–1326) still says "SCOP …
  routes retain Fisher scoring". That has been false since #527. A follow-up, not part of S0.
- **Unchanged since the anatomy:** the QR, dense, QP, structured and MSN code, convergence and
  the dense and structured finalisers, apart from line shifts.

## 2. Block map

Mode codes:
- the steps: **QR**, **SCOP**, **STRUCT** (structured), **DENSE** (gram), **QP** (linear
  constraints);
- **MSN**: the Binomial/log mean space (#425's guard, #437's Newton and its certificate);
- **NEWTON**: the curvature policy;
- **CENTRED**: the centred-system builder and its caches, shared by QR, dense, QP and the
  centred finaliser;
- **CONV**: convergence; **LS**: line search; **INSTR**: instrumentation; **FIN**: finalisation.

"Writers" are the top blame contributors (non-blank lines). "Intent" is the problem the code
solves, from the PRs and the comments at the code.

| ID | Lines | Mode | Purpose and intent | Writers [blame] |
|---|---|---|---|---|
| S01 | 1172–1348 | shared | Signature (47 parameters, 18 of them underscore-private switches) and docstring; one private switch per caller | 38bf9706, #147, #156, #425 |
| S02 | 1349–1373 | shared | Wraps an ndarray X; validates weights; objective merit scale (Poisson/sqrt response mass, else 1) | 38bf9706, #156 |
| S03 | 1374–1426 | STRUCT | Structured backend and layout chosen from structure, never from data (#165, #425); fs BLAS cap; nested/fs state centre c0 so eta never cancels Xβ against a raw intercept (§3.8) | #165, #425 |
| S04 | 1427–1464 | shared+MSN | Offset and start; a default intercept is moved into the mean space (Binomial/log, lowered, #437) or the link domain (raised, #498), since the guards hold only from a feasible start | #437, #498 |
| S05 | 1465–1475 | shared | Dense S, or None on the structured route | #165 |
| S06 | 1476–1536 | shared | Penalty operators without forming S: S·v and its magnitude, rounding count, diag(S), the quadratic, their caches | #165, #424, #425 |
| S07 | 1537–1602 | CONV | `mode_residual`: the certified stop, a relative penalised score centred on the last solve's system (#425), floored at rounding (#439) | #425, #439 |
| S08 | 1603–1957 | MSN | `true_mode_residual`: the same certificate on the unclipped Binomial/log score. It covers the NNLS KKT residual on active rows, light row sets, truncated directions (#454) and the SCOP latent score (joint since #527), with power-of-two equilibration | #437 (298), #454 |
| S09 | 1958–2004 | INSTR | Trace ids, resolved lambdas, `emit_evaluation` | #156 |
| S10 | 2005–2071 | shared | `evaluate_state`: the one constructor of `_IRLSState`; the centred state (§3.8); penalised deviance unless SCOP | #156, #425 |
| S11 | 2072–2113 | INSTR | `emit_state_commit` | #156, #437 |
| S12 | 2114–2136 | QP | Stacks A β ≥ b and \|A\| once per fit (feasibility scale) | #87, #361 |
| S13 | 2137–2162 | SCOP+DENSE | `_has_scop`; `_scop_columns` (read only by S08); the gram/QR state centre (a raw 1e8 column left the merit noisy, #425) | #425, #437 |
| S14 | 2163–2178 | shared | SCOP REML curvature class; refuses incoherent private-switch combinations | #156, #147 |
| S15 | 2179–2237 | NEWTON | Curvature fixed once: observed Newton from the start, fREML working Newton (#500), Gamma/log switchable (#498), the latch; SCOP included since #527 | #498, #425, #500 |
| S15b | 2238–2256 | NEWTON | `observed_rows_invalid`: a trial without finite observed rows is rejected by every observed search, SCOP's included (#527) | #527, #425 |
| S16 | 2257–2443 | SCOP | Per-group SCOP state (θ, basis, penalty, bins, lone exact support through `SCOPRunCentring`); the reduced non-SCOP design shared across a REML run; `SCOPJointNewton`; θ initialised by QP | #88, #147, #527, #93 |
| S17 | 2444–2466 | QR+MSN | QR switch; materialises X and √S; mean-space Newton availability | 48d37cd9, #437 |
| S18 | 2467–2588 | CENTRED | Performance plumbing within ownership rules (#407): tabmat split, constant-weight Gram reuse, raw-spline deferral, refusal seeding, raw-moment policy, Fisher-data and initial-data reuse, exact-type gates | #407, #156 |
| S19 | 2589–2648 | CENTRED | `get_centered_system`: centred normal equations with the intercept profiled, through the caches | #407, #147 |
| S20 | 2649–2714 | CENTRED | `certify_centered_factor`: factors the grouped augmented design when the Gram is not authoritative, cached per immutable geometry | #156, #439 |
| S21 | 2715–2800 | shared | Loop state: timers, last working systems, score centre, ratio histories, bar and window, MSN flags, held-rank and truncation cells, domain predicates, the carried centred warm start | #425, #439, #437, #454 |
| S22 | 2801–2922 | shared+SCOP | Fit-entry committed state; SCOP latent bundle and merit with a forward-error allowance (NB2 rounding since #527); SCOP objective scale 0 | #156, #147, da7047c9, #527 |
| S23 | 2923–2966 | shared | Objective, eta and mu; diagnostics and debug flags; halving budget 20; SVD counter; QP latches (#472); `deviance_relative_change` | #147, 1974498e |
| L01 | 2967–2985 | shared | Per-iteration reset; withdraws the QP certificate, because W changes | #147, 95026013 |
| L02 | 2986–3049 | shared+MSN | Working rows (Fisher or observed, from the policy); MSN rows (#437); refuses non-finite observed entry rows; Fisher weight statistics | #437, #156, #147 |
| L03 | 3050–3082 | QR | QR of [√W X̃; √S]; centred intercept; held rank; score centre | #425, #147, #439 |
| L03b | 3083–3087 | shared | Gram-path prologue (`z_off`) | 48d37cd9 |
| L04 | 3088–3220 | SCOP | Joint Newton step (#527): ordinary solve on the reduced design, `SCOPJointNewton.step`, `_SCOPOrdinarySolve`, the choice of centred prediction, the proposal eta | #527, #156, #465 |
| L05 | 3221–3358 | STRUCT | Structured system and operator; Levenberg shift on a refused observed iterate (§3.11); minimum-norm increment on a truncated sz factor or under MSN (Pes & Rodriguez 2021); score centre | #425, #165, #439 |
| L06 | 3359–3429 | DENSE | Cholesky or certified factor; iterate (Fisher) or increment (MSN); centred intercept; held rank; score centre | #437, #147, #425 |
| L07 | 3430–3483 | QP | Active-set QP with warm start, cold and degenerate-vertex retries (#472, latched); certificate; condition | #477, #437, 0987cb7f |
| L08 | 3484–3499 | shared | Warns after 3 consecutive uncertified SVD fallbacks (not QR) | 48d37cd9 |
| L09 | 3500–3666 | SCOP | SCOP proposal, latent merit, latent maximum step (#527), search with the observed-row predicate, relabel, write-back to `_scop_state` | #147, #156, #527 |
| L10 | 3667–3712 | LS | Proposal state; fixed-endpoint trial evaluator with cached directions | #156, #425 |
| L11 | 3713–3742 | LS+QP | Feasibility of the committed state, the proposal and each trial; the observed-row predicate by default | 0987cb7f |
| L12 | 3743–3770 | MSN | Newton merit on the unclipped log-likelihood (log1mexp) and its rounding | #437 |
| L13 | 3771–3795 | LS | Chains the mean-space and link-domain predicates while the committed state is inside | #425, #498 |
| L14 | 3796–3850 | LS | Halving call: budgets (Poisson/sqrt, lowered start), merit delta, first-halving skip | #147, #437, #454 |
| L15 | 3851–3905 | LS+QP | Accepts a feasible QP proposal when an infeasible committed state rejects every trial; picks the retained state; QP certificate only after a full step | 0987cb7f, #147 |
| L16 | 3906–4067 | CONV | Mode-score stop with stagnation, the Gamma/log switch (#498) and the float64 fixed-point stop; coefficient stop with the SCOP latent change; objective stop; damped SCOP never stops (#527) | #498, #156, #425 |
| L17 | 4068–4149 | MSN | Certifies a stop the clip can fake, on the true score; under Newton the certificate is the stop; switches to Newton when refused near the mode or stalled | #437 |
| L18 | 4150–4200 | QP+CONV | Feasibility and KKT veto; termination-reason chain | 0987cb7f, da400514, #156 |
| L19 | 4201–4237 | INSTR | Trace step decision and state commit | #156 |
| L20 | 4238–4366 | INSTR | Extrema, `IterationDiagnostics`, `deviance_relative_change` (F02 reads it), log, debug rows | 17a9fae2, #147, #105 |
| L21 | 4367–4388 | shared | Break on rejection; commit; break on convergence or stagnation; advance | #147, #156 |
| F00 | 4389–4391 | INSTR | Done log | 38bf9706 |
| F01 | 4392–4418 | FIN | Gaussian/identity compensated centred intercept (α, α_lo) and the deviance there | #425 |
| F02 | 4419–4472 | FIN | Runtime separation backstop | 2404aa94, #422 |
| F03 | 4473–4493 | QP | Final refusal of an infeasible or KKT-incomplete state | 0987cb7f, da400514 |
| F04 | 4494–4550 | MSN | Rows at the clip boundary are never converged; truncated directions published; boundary relabel | #454, #425 |
| F05 | 4551–4581 | SCOP | Terminal SCOP curvature at the retained model (`block_curvature`, no step; #527) | #527 |
| F06 | 4582–4597 | FIN | Fisher export rows | #156 |
| F07 | 4598–4635 | INSTR | Profile timings and the stop rule's numbers | #425, #437 |
| F08 | 4636–4702 | FIN | Final structured or centred system and its moments | #165 |
| F09 | 4703–4750 | FIN | `cache_out` for cached-W fREML | #165, #156 |
| F10 | 4751–4828 | STRUCT | Profiled structured factor, log det H, rank, EDF | #165, #424 |
| F11 | 4829–4987 | DENSE | Pseudo-inverse, log det H, REML summary, EDF, `RankInfo` inline or deferred (#492) | #147, #492, #494 |
| F12 | 4988–5020 | INSTR | Finalisation time; backend disclosure | #425, #165 |
| F13 | 5021–5042 | FIN | Pearson φ; NaN for fits that skip statistics | #362, 3913dbe7 |
| F14 | 5043–5170 | FIN+SCOP | `PIRLSResult`; SCOP post-fit inference and φ; four return shapes | #156, #90 |

## 3. Fields per block

[AST] A field is a function-scope name, parameters included, that more than one block touches;
there are 212. The tables include the reused temporaries (`g`, `gi`, `st`, `centered`, `z_off`, `group`, `certification`, `certified`), which stay method locals. Each row lists:
- **Reads (bound elsewhere):** the fields the block reads but does not bind.
- **Writes:** the fields it binds.
- **Mutates:** the fields it changes in place (item or attribute assignment, `append`,
  `clear`, `update`).

Captures inside a nested `def` are listed in §4, not here.

| ID | Lines | Lines/code | Writers | Reads (bound elsewhere) | Writes | Mutates |
|---|---|---|---|---|---|---|
| S01 | 1172–1348 | 177/50 | 38bf9706 53, #147 20, #156 19, #425 14 | — | — | — |
| S02 | 1349–1373 | 25/21 | 38bf9706 8, #156 7, #147 4, c77876c3 2 | `family`, `groups`, `link`, `y` | `dm`, `gms`, `n`, `objective_merit_scale`, `p`, `weights` | — |
| S03 | 1374–1426 | 53/45 | #165 32, #425 19, #424 1 | `S_override`, `direct_solve`, `dm`, `gms`, `groups`, `lambda2`, `p`, `reml_penalties`, `weights` | `_direct_fallback_reason`, `_state_center`, `_structured_group_index`, `_structured_layout`, `_use_structured`, `structured_decision` | `_state_center` |
| S04 | 1427–1464 | 38/28 | #437 12, #498 11, 38bf9706 6, c77876c3 3 | `beta_init`, `dm`, `family`, `intercept_init`, `link`, `n`, `p`, `weights`, `y` | `_deviance_init`, `_start_lowered`, `beta`, `intercept`, `offset` | — |
| S05 | 1465–1475 | 11/7 | #165 7, #75 3 | `S_override`, `_use_structured`, `gms`, `groups`, `lambda2`, `p`, `reml_penalties` | `S` | — |
| S06 | 1476–1536 | 61/42 | #165 17, #424 14, #425 13, #465 4 | — | `penalty_matvec` | — |
| S07 | 1537–1602 | 66/52 | #425 33, #439 15, #156 5, #424 4 | — | `mode_residual` | — |
| S08 | 1603–1957 | 355/262 | #437 298, #454 16, #425 11, #527 7 | — | `true_mode_residual` | — |
| S09 | 1958–2004 | 47/45 | #156 45 | `lambda2`, `trace_run` | `emit_evaluation`, `trace_enabled` | — |
| S10 | 2005–2071 | 67/62 | #156 43, #425 16, #147 6, #165 1 | — | `evaluate_state` | — |
| S11 | 2072–2113 | 42/41 | #156 32, #437 8, 770e295b 1 | — | `emit_state_commit` | — |
| S12 | 2114–2136 | 23/17 | #87 13, #361 6, 015aa891 3 | `groups`, `p` | `A_all`, `abs_A_all`, `b_all`, `g`, `has_constraints`, `prev_active_set` | — |
| S13 | 2137–2162 | 26/16 | #425 13, #437 7, #88 3, #156 2 | `_use_structured`, `dm`, `groups`, `has_constraints`, `p`, `weights` | `_has_scop`, `_scop_run`, `_state_center`, `g` | — |
| S14 | 2163–2178 | 16/15 | #156 9, #147 4, #492 2 | `_compute_fit_statistics`, `_compute_reml_geometry`, `_defer_rank_info`, `_has_scop`, `_return_working_system`, `compute_rank_info`, `family`, `has_constraints`, `link` | `_scop_curvature` | — |
| S15 | 2179–2237 | 59/20 | #498 22, #425 10, #500 9, #156 7 | `_has_scop`, `_newton_latch`, `_return_working_system`, `family`, `has_constraints`, `link` | `_observed_newton_active`, `_stagnation_newton_available` | — |
| S15b | 2238–2256 | 19/11 | #527 7, #425 6, #156 3, #424 1 | — | `observed_rows_invalid` | — |
| S16 | 2257–2443 | 187/172 | #88 35, #147 33, #527 28, #93 23 | `S`, `_has_scop`, `beta`, `debug_recorder`, `dm`, `evaluate_state`, `family`, `gms`, `groups`, `intercept`, `lambda2`, `link`, `n`, `offset`, `trace_run`, `weights`, `y` | `_S_reduced`, `_expose_exact_support_state`, `_reduced_dm`, `_reduced_tabmat_state`, `_reduced_to_full`, `_scop_newton`, `_scop_order`, `_scop_run`, `_scop_specs`, `_scop_state`, `g`, `gi`, `st` | `_scop_state`, `beta`, `st` |
| S17 | 2444–2466 | 23/20 | 48d37cd9 10, #437 4, 16308f4a 4, #88 2 | `S`, `_has_scop`, `_return_working_system`, `direct_solve`, `family`, `gms`, `has_constraints`, `link`, `p` | `_L_aug`, `_X_full`, `_mean_space_newton_available`, `_use_qr`, `gm` | — |
| S18 | 2467–2588 | 122/107 | #407 73, #156 30, #165 7, 0987cb7f 3 | `_has_scop`, `_observed_newton_active`, `_scop_run`, `_use_qr`, `_use_structured`, `beta`, `debug_recorder`, `dm`, `family`, `gms`, `groups`, `has_constraints`, `intercept`, `link`, `n`, `profile`, `trace_purpose`, `trace_run` | `_can_reuse_weighted_gram`, `gm`, `group` | `profile` |
| S19 | 2589–2648 | 60/59 | #407 27, #147 24, #156 2, #465 2 | — | `get_centered_system` | — |
| S20 | 2649–2714 | 66/57 | #156 57, #439 5, #407 2 | — | `certify_centered_factor` | — |
| S21 | 2715–2800 | 86/48 | #425 21, #439 19, #437 17, #454 8 | `_state_center`, `beta`, `beta_init`, `family`, `intercept`, `intercept_init`, `link`, `max_iter` | `XtWX_beta`, `_final_penalized_operator`, `_held_rank`, `_judged_truncated`, `_last_mode_residual`, `_last_true_residual`, `_last_truncated`, `_last_working_centered`, `_last_working_offset_mean`, `_last_working_structured`, `_link_domain_invalid`, `_mean_space_invalid`, `_mean_space_newton`, `_mode_ratios`, `_score_centre`, `_stagnation_window`, `_t_deviance`, `_t_deviance_eval`, `_t_eta`, `_t_gram`, `_t_solve`, `_t_working`, `_true_confirmed`, `_true_ratios`, `_true_stop`, `_true_stop_ratio`, `centred_start`, `converged`, `mode_bar`, `score_stagnated`, `t_start` | — |
| S22 | 2801–2922 | 122/94 | #156 37, #147 24, da7047c9 24, #527 18 | `S`, `_deviance_init`, `_has_scop`, `_scop_specs`, `_scop_state`, `beta`, `centred_start`, `emit_evaluation`, `emit_state_commit`, `evaluate_state`, `family`, `intercept`, `weights`, `y` | `committed`, `gi`, `objective_merit_scale`, `scop_committed`, `st`, `with_scop_merit` | — |
| S23 | 2923–2966 | 44/22 | #147 9, 1974498e 8, 770e295b 7, #156 4 | `committed`, `debug_recorder`, `has_constraints`, `record_diagnostics` | `_consecutive_svd`, `_degeneracy_retry_failed`, `_reported_qp_nonconvergence`, `base_debug_context`, `capture_extrema`, `dev_prev`, `deviance_relative_change`, `eta`, `eta_unclipped`, `iteration_log`, `max_halving`, `mu`, `objective_prev`, `record_debug_rows`, `retained_qp_converged`, `termination_reason` | — |
| L01 | 2967–2985 | 19/13 | #147 7, 95026013 6, #11 1, #437 1 | `_held_rank`, `committed`, `has_constraints`, `max_iter`, `prev_active_set` | `beta`, `beta_prev`, `committed_active_set`, `intercept`, `intercept_prev`, `it`, `proposal_centred_intercept`, `proposal_qp_converged`, `rank_truncated`, `retained_qp_converged`, `scop_proposal_eta_unclipped`, `used_rank_certification` | `_held_rank` |
| L02 | 2986–3049 | 64/51 | #437 26, #156 13, #147 11, #425 9 | `_mean_space_newton`, `_observed_newton_active`, `committed`, `eta`, `eta_unclipped`, `family`, `it`, `link`, `mu`, `profile`, `weights`, `y` | `W`, `_t0`, `_t_working`, `newton_score`, `w_ratio`, `working_eta`, `working_eta_unclipped`, `working_mu`, `working_rows`, `z` | `profile` |
| L03 | 3050–3082 | 33/32 | #425 10, #147 9, #439 5, 48d37cd9 4 | `W`, `_L_aug`, `_X_full`, `_held_rank`, `_state_center`, `_use_qr`, `convergence`, `dm`, `get_centered_system`, `offset`, `p`, `z` | `_cond_est`, `_last_working_centered`, `_last_working_offset_mean`, `_score_centre`, `_t0`, `_t_solve`, `_used_svd`, `beta`, `centered`, `intercept`, `iteration_rank`, `proposal_centred_intercept`, `rank_truncated`, `z_off` | `_held_rank` |
| L03b | 3083–3087 | 5/3 | 48d37cd9 2, #3 1, 38bf9706 1 | `offset`, `z` | `_t0`, `z_off` | — |
| L04 | 3088–3220 | 133/103 | #527 57, #156 38, #465 20, #88 5 | `W`, `_S_reduced`, `_has_scop`, `_reduced_dm`, `_reduced_tabmat_state`, `_reduced_to_full`, `_scop_newton`, `_scop_order`, `_scop_run`, `_scop_specs`, `_scop_state`, `base_debug_context`, `committed`, `debug_recorder`, `groups`, `it`, `offset`, `p`, `profile`, `scop_committed`, `z`, `z_off` | `_cond_est`, `_t0`, `_t_gram`, `_t_solve`, `_used_svd`, `beta`, `centred_level`, `certified`, `gi`, `intercept`, `rank_truncated`, `scop_ordinary`, `scop_proposal_eta_unclipped`, `scop_results`, `used_rank_certification` | `beta` |
| L05 | 3221–3358 | 138/100 | #425 78, #165 28, #439 18, #437 10 | `S_override`, `W`, `_state_center`, `_structured_group_index`, `_structured_layout`, `_use_structured`, `committed`, `convergence`, `dm`, `eta`, `gms`, `groups`, `lambda2`, `newton_score`, `p`, `penalty_matvec`, `profile`, `reml_penalties`, `weights`, `working_rows`, `z`, `z_off` | `_cond_est`, `_final_penalized_operator`, `_last_working_structured`, `_score_centre`, `_t0`, `_t_gram`, `_t_solve`, `_used_svd`, `augmented_factor`, `beta`, `intercept`, `proposal_centred_intercept`, `rank_truncated` | `profile` |
| L06 | 3359–3429 | 71/59 | #437 26, #147 12, #425 12, #156 10 | `W`, `_held_rank`, `_state_center`, `certify_centered_factor`, `committed`, `convergence`, `dm`, `get_centered_system`, `has_constraints`, `newton_score`, `penalty_matvec`, `z_off` | `_cond_est`, `_last_working_centered`, `_last_working_offset_mean`, `_score_centre`, `_t0`, `_t_gram`, `_t_solve`, `_used_svd`, `beta`, `centered`, `certification`, `certified`, `intercept`, `iteration_rank`, `proposal_centred_intercept`, `rank_truncated`, `used_rank_certification` | `_held_rank` |
| L07 | 3430–3483 | 54/36 | #477 8, #437 6, 0987cb7f 6, da400514 6 | `A_all`, `W`, `b_all`, `dm`, `get_centered_system`, `it`, `newton_score`, `z_off` | `_cond_est`, `_degeneracy_retry_failed`, `_last_working_centered`, `_reported_qp_nonconvergence`, `_t0`, `_t_gram`, `_t_solve`, `_used_svd`, `beta`, `centered`, `intercept`, `prev_active_set`, `proposal_qp_converged` | — |
| L08 | 3484–3499 | 16/12 | 48d37cd9 10, #147 5 | `_cond_est`, `_used_svd`, `direct_solve`, `used_rank_certification` | `_consecutive_svd` | — |
| L09 | 3500–3666 | 167/159 | #147 83, #156 36, #527 29, 9d3be519 8 | `_has_scop`, `_observed_newton_active`, `_scop_specs`, `_scop_state`, `centred_level`, `emit_evaluation`, `evaluate_state`, `it`, `max_halving`, `observed_rows_invalid`, `scop_committed`, `scop_ordinary`, `scop_proposal_eta_unclipped`, `scop_results`, `with_scop_merit` | `_t0`, `_t_deviance`, `_t_deviance_eval`, `beta`, `decision`, `dev`, `eta`, `eta_unclipped`, `evaluation_elapsed`, `gi`, `group`, `intercept`, `mu`, `n_halvings`, `proposal_scop`, `retained`, `retained_scop`, `st`, `step_rejected` | `st` |
| L10 | 3667–3712 | 46/44 | #156 25, #425 10, #147 8, bff1a21f 1 | `beta`, `evaluate_state`, `intercept`, `it`, `proposal_centred_intercept` | `_t0`, `evaluate_trial`, `proposal`, `trial_cache` | — |
| L11 | 3713–3742 | 30/28 | 0987cb7f 24, #361 3, #527 1 | `A_all`, `_observed_newton_active`, `abs_A_all`, `b_all`, `committed`, `has_constraints`, `observed_rows_invalid`, `proposal` | `committed_constraints_feasible`, `constraint_trial_is_invalid`, `proposal_constraints_feasible` | — |
| L12 | 3743–3770 | 28/17 | #437 24 | — | `_newton_merit_roundoff` | — |
| L13 | 3771–3795 | 25/19 | #425 17, #498 6 | `_link_domain_invalid`, `_mean_space_invalid`, `committed`, `constraint_trial_is_invalid`, `weights` | `trial_is_invalid` | — |
| L14 | 3796–3850 | 55/49 | #147 21, #437 15, #454 8, c77876c3 4 | `_newton_merit_roundoff`, `committed`, `evaluate_trial`, `max_halving`, `newton_score`, `objective_merit_scale`, `profile`, `proposal`, `trial_is_invalid` | `decision` | `profile` |
| L15 | 3851–3905 | 55/45 | 0987cb7f 22, #147 22, 95026013 7, #156 1 | `_t0`, `committed`, `committed_active_set`, `committed_constraints_feasible`, `has_constraints`, `it`, `proposal`, `proposal_constraints_feasible`, `proposal_qp_converged`, `trial_cache` | `_t_deviance`, `_t_deviance_eval`, `beta`, `decision`, `dev`, `eta`, `eta_unclipped`, `evaluation_elapsed`, `intercept`, `mu`, `n_halvings`, `prev_active_set`, `retained`, `retained_qp_converged`, `step_rejected` | — |
| L16 | 3906–4067 | 162/121 | #498 67, #156 32, #425 29, #527 9 | `_has_scop`, `_mode_ratios`, `_newton_latch`, `_score_centre`, `_stagnation_newton_available`, `_stagnation_window`, `beta`, `beta_prev`, `committed`, `convergence`, `dev`, `eta`, `has_constraints`, `intercept`, `intercept_prev`, `it`, `max_iter`, `mode_bar`, `mode_residual`, `mu`, `n_halvings`, `objective_merit_scale`, `objective_prev`, `profile`, `proposal`, `proposal_scop`, `retained`, `retained_scop`, `scop_committed`, `step_rejected` | `_can_reuse_weighted_gram`, `_last_mode_residual`, `_observed_newton_active`, `coef_change`, `converged_this_iter`, `convergence_label`, `convergence_tolerance`, `convergence_value`, `dev_rel_change`, `proposal_state`, `score_stagnated` | `_mode_ratios`, `_newton_latch`, `profile` |
| L17 | 4068–4149 | 82/51 | #437 79, #147 1 | `A_all`, `_has_scop`, `_mean_space_invalid`, `_mean_space_newton_available`, `_stagnation_window`, `_start_lowered`, `_true_ratios`, `committed`, `dev`, `family`, `has_constraints`, `link`, `mode_bar`, `prev_active_set`, `retained`, `retained_scop`, `step_rejected`, `true_mode_residual`, `weights` | `_mean_space_newton`, `_true_confirmed`, `_true_stop`, `_true_stop_ratio`, `converged_this_iter`, `convergence_label`, `convergence_tolerance`, `convergence_value`, `score_stagnated` | `_true_ratios` |
| L18 | 4150–4200 | 51/40 | 0987cb7f 22, da400514 10, #156 10, #425 3 | `A_all`, `abs_A_all`, `b_all`, `beta`, `dev`, `has_constraints`, `it`, `max_iter`, `retained_qp_converged`, `score_stagnated`, `step_rejected` | `converged_this_iter`, `termination_reason` | — |
| L19 | 4201–4237 | 37/36 | #156 29, #437 7 | `converged_this_iter`, `convergence_label`, `convergence_tolerance`, `convergence_value`, `dev`, `emit_state_commit`, `it`, `retained`, `termination_reason`, `trace_enabled`, `trace_purpose`, `trace_run` | — | — |
| L20 | 4238–4366 | 129/119 | 17a9fae2 31, #147 26, #105 24, #156 14 | `W`, `_cond_est`, `_has_scop`, `_used_svd`, `base_debug_context`, `capture_extrema`, `coef_change`, `committed`, `converged_this_iter`, `convergence`, `convergence_label`, `convergence_tolerance`, `convergence_value`, `debug_recorder`, `decision`, `dev`, `dev_prev`, `dev_rel_change`, `eta`, `eta_unclipped`, `intercept`, `it`, `iteration_log`, `mu`, `n_halvings`, `newton_score`, `proposal_state`, `rank_truncated`, `record_debug_rows`, `record_diagnostics`, `retained`, `step_rejected`, `termination_reason`, `w_ratio`, `working_eta`, `working_eta_unclipped`, `working_mu`, `working_rows` | `deviance_relative_change` | `iteration_log` |
| L21 | 4367–4388 | 22/20 | #147 8, #156 5, #425 3, 38bf9706 3 | `_has_scop`, `converged_this_iter`, `dev`, `it`, `retained`, `retained_scop`, `score_stagnated`, `step_rejected` | `committed`, `converged`, `dev_prev`, `objective_prev`, `scop_committed` | — |
| F00 | 4389–4391 | 3/2 | 38bf9706 2 | `it`, `t_start` | `t_elapsed` | — |
| F01 | 4392–4418 | 27/19 | #425 24, #156 1, #477 1 | `_compute_fit_statistics`, `_state_center`, `beta`, `dm`, `family`, `link`, `offset`, `retained`, `weights`, `y` | `centred_intercept_lo`, `dev`, `eta`, `eta_unclipped`, `mu` | — |
| F02 | 4419–4472 | 54/33 | 2404aa94 32, #422 9, 5e475e48 5, #487 2 | `beta`, `converged`, `dev`, `deviance_relative_change`, `dm`, `eta`, `eta_unclipped`, `family`, `groups`, `it`, `link`, `max_iter`, `step_rejected`, `w_ratio`, `weights`, `working_eta`, `working_mu`, `working_rows` | — | — |
| F03 | 4473–4493 | 21/20 | 0987cb7f 12, da400514 7, #361 1 | `A_all`, `abs_A_all`, `b_all`, `beta`, `has_constraints`, `retained_qp_converged` | `converged`, `termination_reason` | — |
| F04 | 4494–4550 | 57/39 | #454 38, #425 18 | `_judged_truncated`, `_last_truncated`, `_mean_space_invalid`, `eta_unclipped`, `family`, `link`, `weights` | `_boundary_rows`, `_published_truncated`, `_truncated_boundary`, `converged`, `termination_reason` | — |
| F05 | 4551–4581 | 31/22 | #527 13, #156 6, c77876c3 5, #147 4 | `_has_scop`, `_scop_newton`, `_scop_order`, `_scop_state`, `eta`, `eta_unclipped`, `family`, `link`, `mu`, `weights`, `y` | `W`, `gi`, `z` | `_scop_state` |
| F06 | 4582–4597 | 16/12 | #156 6, c77876c3 5, #147 4 | `_return_working_system`, `eta`, `family`, `link`, `mu`, `weights`, `y` | `W`, `export_rows`, `z` | — |
| F07 | 4598–4635 | 38/32 | #425 14, #437 10, #3 9, bff1a21f 4 | `_boundary_rows`, `_last_mode_residual`, `_last_true_residual`, `_t_deviance`, `_t_deviance_eval`, `_t_eta`, `_t_gram`, `_t_solve`, `_t_working`, `_true_stop`, `_true_stop_ratio`, `it`, `mode_bar`, `profile`, `t_elapsed` | — | `profile` |
| F08 | 4636–4702 | 67/63 | #165 44, #147 10, #439 6, #425 3 | `S_override`, `W`, `_last_working_centered`, `_last_working_offset_mean`, `_last_working_structured`, `_return_working_system`, `_state_center`, `_structured_group_index`, `_structured_layout`, `_use_structured`, `cache_out`, `dm`, `export_rows`, `get_centered_system`, `gms`, `groups`, `lambda2`, `offset`, `p`, `reml_penalties`, `weights`, `z` | `XtW1`, `XtWX`, `XtWz`, `_final_penalized_operator`, `centered_final`, `centered_rhs`, `mean_x`, `mean_z`, `offset_mean_final`, `structured_final`, `sum_W`, `sum_Wz`, `z_off` | `XtW1`, `XtWz` |
| F09 | 4703–4750 | 48/31 | #165 14, #156 10, #425 9, #3 5 | `W`, `XtW1`, `XtWX`, `XtWz`, `_final_penalized_operator`, `_has_scop`, `_return_working_system`, `_use_structured`, `cache_out`, `centered_final`, `centered_rhs`, `dm`, `eta_unclipped`, `mean_x`, `mean_z`, `offset_mean_final`, `structured_final`, `sum_W`, `sum_Wz` | — | `cache_out` |
| F10 | 4751–4828 | 78/68 | #165 40, #424 12, #156 10, #425 10 | `XtW1`, `_compute_fit_statistics`, `_compute_reml_geometry`, `_final_penalized_operator`, `_use_structured`, `mean_x`, `structured_final`, `sum_W` | `XtWX_S_inv_beta`, `XtWX_beta`, `_t0`, `augmented_factor`, `log_det_H`, `p_eff`, `rank_info`, `reml_geometry_summary`, `reml_hessian_rank`, `retained_reml_decomposition`, `structured_factor` | — |
| F11 | 4829–4987 | 159/125 | #147 37, #492 28, #156 26, #494 18 | `W`, `XtWX`, `_compute_fit_statistics`, `_compute_reml_geometry`, `_defer_rank_info`, `_use_qr`, `centered_final`, `certify_centered_factor`, `compute_rank_info` | `XtWX_S_inv_beta`, `XtWX_beta`, `certification`, `log_det_H`, `p_eff`, `rank_info`, `reml_geometry_summary`, `reml_hessian_rank`, `retained_reml_decomposition` | `_defer_rank_info` |
| F12 | 4988–5020 | 33/28 | #425 12, #165 8, #424 6, #3 3 | `_direct_fallback_reason`, `_t0`, `_use_qr`, `_use_structured`, `direct_solve`, `groups`, `profile`, `structured_decision`, `structured_factor` | `_resolved_direct_backend` | `profile` |
| F13 | 5021–5042 | 22/13 | #362 8, 3913dbe7 6, #156 3, da7047c9 2 | `_compute_fit_statistics`, `_compute_reml_geometry`, `family`, `mu`, `weight_semantics`, `weights`, `y` | `p_eff`, `pearson_sum`, `phi` | — |
| F14 | 5043–5170 | 128/118 | #156 64, #90 16, 38bf9706 12, #105 6 | `S`, `W`, `XtW1`, `XtWX`, `XtWX_S_inv_beta`, `XtWX_beta`, `_boundary_rows`, `_compute_fit_statistics`, `_direct_fallback_reason`, `_expose_exact_support_state`, `_has_scop`, `_published_truncated`, `_resolved_direct_backend`, `_scop_curvature`, `_scop_state`, `_state_center`, `_true_confirmed`, `_truncated_boundary`, `beta`, `centered_final`, `centred_intercept_lo`, `converged`, `dev`, `dm`, `eta_unclipped`, `family`, `groups`, `intercept`, `it`, `iteration_log`, `link`, `log_det_H`, `mu`, `offset`, `p_eff`, `phi`, `rank_info`, `record_diagnostics`, `reml_geometry_summary`, `reml_hessian_rank`, `retained`, `retained_reml_decomposition`, `sum_W`, `termination_reason`, `weight_semantics`, `weights`, `y` | `gi`, `pearson_sum`, `st` | — |

The function rebinds five of its parameters: `weights` (S02), `offset` (S04), `_deviance_init`
(S04), `_raw_moment_policy` (S18) and `_fisher_data_reuse` (S18). [AST]

## 4. Closures: what each captures, and late binding

[AST] For every nested `def` and lambda (generator expressions are left out):
- **Captures:** the fit-scope names it reads.
- **Called at:** its call sites.
- **Nonlocal:** what it rebinds with `nonlocal`.
- **Late:** the captured names that are first bound, or rebound, after the `def`. The closure
  reads those at call time.

Parameters that are never rebound are not listed.

| Closure | Def line | Inside | Captures | Called at | Nonlocal | Late |
|---|---|---|---|---|---|---|
| `penalty_matvec` | 1476 | fit | `S`, `gms`, `lambda2`, `reml_penalties` | 1535, 1593, 1594, 1680, 1681, 1757 | — | — |
| `penalty_rounding` | 1507 | fit | `S`, `p`, `reml_penalties` | 1600, 1780, 1796, 1818, 1854 | — | — |
| `penalty_curvature` | 1513 | fit | `S`, `lambda2`, `p`, `reml_penalties` | 1583, 1679 | — | — |
| `penalty_quadratic` | 1533 | fit | `penalty_matvec` | 2057 | — | — |
| `mode_residual` | 1537 | fit | `_laplace_excluded`, `_penalty_curvature`, `_score_centre`, `dm`, `family`, `link`, `mode_bar`, `offset`, `p`, `penalty_curvature`, `penalty_matvec`, `penalty_rounding`, `weights`, `y` | 3921, 4000 | — | `_score_centre` bound only after the def, at 2743, 3075, 3355, 3422; `mode_bar` bound only after the def, at 2750 |
| `true_mode_residual` | 1603 | fit | `_held_rank`, `_judged_truncated`, `_laplace_excluded`, `_last_true_residual`, `_last_truncated`, `_penalty_curvature`, `_row_set_curvature`, `_scop_columns`, `_scop_state`, `dm`, `groups`, `lambda2`, `mode_bar`, `n`, `offset`, `p`, `penalty_curvature`, `penalty_matvec`, `penalty_rounding`, `profile`, `weights`, `y` | 4101 | — | `_held_rank` bound only after the def, at 2775; `_judged_truncated` bound only after the def, at 2776; `_last_true_residual` bound only after the def, at 2768; `_last_truncated` bound only after the def, at 2777; `_scop_columns` bound only after the def, at 2142, 2144; `_scop_state` bound only after the def, at 2260; `mode_bar` bound only after the def, at 2750 |
| `likelihood_noise` | 1732 | true_mode_residual | `weights`, `y` | passed as a value | — | — |
| `weak_penalty_block` | 1750 | true_mode_residual | `p`, `penalty_matvec` | passed as a value | — | — |
| `lambda` | 1846 | true_mode_residual | `penalty_matvec` | passed as a value | — | — |
| `lambda` | 1851 | true_mode_residual | `penalty_matvec` | passed as a value | — | — |
| `by_bin` | 1891 | true_mode_residual | — | 1901, 1902, 1905 | — | — |
| `emit_evaluation` | 1969 | fit | `trace_enabled`, `trace_purpose`, `trace_run` | 2062, 2910, 3541, 3575 | — | — |
| `lambda` | 1983 | emit_evaluation | — | passed as a value | — | — |
| `evaluate_state` | 2005 | fit | `_has_scop`, `_state_center`, `dm`, `emit_evaluation`, `family`, `link`, `offset`, `penalty_quadratic`, `resolved_lambdas`, `trace_basis_id`, `trace_enabled`, `trace_run`, `weights`, `y` | 2394, 2802, 3505, 3669, 3700 | — | `_has_scop` bound only after the def, at 2138; `_state_center` rebound after the def, at 2162 |
| `emit_state_commit` | 2072 | fit | `convergence`, `tol`, `trace_enabled`, `trace_purpose`, `trace_run` | 2916, 4228 | — | — |
| `lambda` | 2087 | emit_state_commit | `convergence`, `tol` | passed as a value | — | — |
| `observed_rows_invalid` | 2238 | fit | `family`, `link`, `weights`, `y` | passed as a value | — | — |
| `get_centered_system` | 2589 | fit | `S`, `_can_reuse_weighted_gram`, `_constant_centered_cache`, `_constant_centered_z`, `_fisher_data_reuse`, `_initial_data_key`, `_initial_data_pending`, `_initial_data_reuse`, `_observed_newton_active`, `_raw_moment_policy`, `_scop_run`, `_tabmat_centering_state`, `_tabmat_split`, `dm`, `fixed_owners`, `profile`, `weight_semantics` | 3055, 3360, 3431, 4691 | `_constant_centered_cache`, `_constant_centered_z`, `_initial_data_pending` | `_can_reuse_weighted_gram` rebound after the def, at 3960; `_observed_newton_active` rebound after the def, at 3959 |
| `certify_centered_factor` | 2649 | fit | `_centered_factor_certification`, `_fisher_data_reuse`, `dm` | 3377, 4837, 4908, 4929 | `_centered_factor_certification` | — |
| `with_scop_merit` | 2868 | fit | `_scop_specs`, `family`, `gaussian_response_roundoff`, `groups`, `lambda2`, `merit_gamma`, `nb2_unit_roundoff`, `scop_merit_errors`, `scop_outer_penalty`, `scop_outer_penalty_abs` | 2905, 3539, 3574 | — | — |
| `evaluate_scop_trial` | 3549 | fit | `_scop_specs`, `dm`, `emit_evaluation`, `family`, `it`, `link`, `offset`, `proposal_scop`, `resolved_lambdas`, `scop_committed`, `scop_ordinary`, `scop_trial_cache`, `trace_basis_id`, `trace_enabled`, `trace_run`, `weights`, `with_scop_merit`, `y` | 3590, 3594 | — | `scop_committed` rebound after the def, at 4376 |
| `lambda` | 3594 | fit | `evaluate_scop_trial`, `shortened` | passed as a value | — | — |
| `lambda` | 3597 | fit | `family`, `link`, `max_halving`, `scop_committed`, `search_proposal`, `weights`, `y` | passed as a value | — | `scop_committed` rebound after the def, at 4376 |
| `lambda` | 3607 | fit | `scop_merit_errors` | passed as a value | — | — |
| `evaluate_trial` | 3680 | fit | `committed`, `evaluate_state`, `it`, `proposal`, `trial_cache`, `trial_directions` | passed as a value | `trial_directions` | `committed` rebound after the def, at 4374 |
| `constraint_trial_is_invalid` | 3734 | fit | `A_all`, `abs_A_all`, `b_all` | passed as a value | — | — |
| `_newton_rows_at` | 3747 | fit | `_trial_rows`, `committed`, `newton_committed_rows`, `weights`, `y` | 3757, 3757, 3764, 3765 | — | `committed` rebound after the def, at 4374 |
| `_newton_deviance_delta` | 3755 | fit | `_newton_rows_at` | 3834 | — | — |
| `_newton_merit_roundoff` | 3759 | fit | `_newton_rows_at` | passed as a value | — | — |
| `trial_is_invalid` | 3777 | fit | `_mean_space_invalid`, `weights` | passed as a value | — | — |
| `trial_is_invalid` | 3790 | fit | `_link_domain_invalid`, `weights` | passed as a value | — | — |
| `lambda` | 3802 | fit | `_start_lowered`, `committed`, `family`, `link`, `max_halving`, `proposal`, `weights`, `y` | passed as a value | — | `committed` rebound after the def, at 4374 |
| `lambda` | 3828 | fit | `_newton_deviance_delta`, `newton_score`, `penalty_matvec` | passed as a value | — | — |
| `lambda` | 3839 | fit | `committed`, `family`, `link`, `proposal`, `weights` | passed as a value | — | `committed` rebound after the def, at 4374 |
| `lambda` | 4205 | fit | `committed`, `converged_this_iter`, `convergence_label`, `convergence_tolerance`, `convergence_value`, `decision`, `it`, `newton_score`, `proposal_state`, `retained`, `termination_reason`, `working_rows` | passed as a value | — | `committed` rebound after the def, at 4374; `termination_reason` rebound after the def, at 4479, 4488, 4505, 4543 |
| `lambda` | 4870 | fit | `W`, `dm`, `final_system` | passed as a value | — | — |
| `retained_rank_info` | 4880 | fit | `_X_full`, `_return_working_system`, `_use_qr`, `certify_centered_factor`, `dm`, `groups`, `p` | 4980 | — | — |
| `data_factor_rank` | 4927 | retained_rank_info | `certify_centered_factor`, `dm` | passed as a value | — | — |
| `lambda` | 4945 | retained_rank_info | `dm` | passed as a value | — | — |
| `lambda` | 4948 | retained_rank_info | `dm` | passed as a value | — | — |

**What the late bindings mean for S1.**

- **`mode_residual`** reads `_score_centre` and `mode_bar` at call time.
  - `mode_bar` is first bound at 2750.
  - `_score_centre` is bound at 2743 and rebound by QR (3075), structured (3355) and dense
    (3422) in every iteration, before L16 calls it (3921, 4000).
  - As a method, it must read the current field, never a value cached in a local before the
    step.
- **`true_mode_residual`** reads seven names that are first bound after its `def`:
  `_scop_columns` (2142–2144), `_scop_state` (2260), `mode_bar` (2750), `_last_true_residual`
  (2768), `_held_rank` (2775), `_judged_truncated` (2776) and `_last_truncated` (2777).
  - Its only call is at 4101 (L17).
  - The four cells are one-element lists mutated in place: `_held_rank[0]` is reset in L01 and
    written in L03 and L06.
- **`evaluate_state`** reads `_has_scop`, first bound at 2138, and `_state_center`, bound in
  S03 (1419, 1424) and rebound in S13 (2162). Its first call is at 2394, after both, so the fit
  object must bind `state_center` before any evaluation.
- **`get_centered_system`** reads `_observed_newton_active` and `_can_reuse_weighted_gram`, which
  the Gamma/log switch rebinds at 3959–3960.
  - After the switch, the next iteration's centred system must skip the constant-weight cache
    and the Fisher-data reuse.
  - So the builder must read the policy's current state, or be told through
    `disable_weighted_gram_reuse()`.
- **The per-iteration closures** are created again in every iteration: `evaluate_trial`,
  `_newton_rows_at`, `evaluate_scop_trial`, the halving lambdas and the trace lambda. They read
  `committed` or `scop_committed`, which L21 rebinds at 4374/4376 after the closures' last call.
  - Hoisting any of them out of the loop needs `committed` passed explicitly.
  - `trial_directions` (a `nonlocal` of `evaluate_trial`) and `_trial_rows` must start empty in
    every iteration.
- **`trial_is_invalid` (3777, 3790)** binds the previous predicate through a default argument
  (`other=…`), which is early binding. The chain's order is link domain, then mean space, then
  constraint or observed rows. The result does not depend on that order.
- **`retained_rank_info` (4880).**
  - Bound by default arguments: `centered_final`, `W`, `augmented_rank` and `feature_edf`.
  - Read at call time: `dm`, `groups`, `p`, `_use_qr`, `_X_full`, `_return_working_system` and
    `certify_centered_factor`.
  - It may run after the solve returns (`_defer_rank_info`; 22 deferred calls in 6 harness
    cases, NB2 with θ auto).
- **The trace lambdas** (1983, 2087, 4205) are consumed at once: `TraceRun.emit_lazy` calls the
  factory before it returns (`_fit_trace.py:276`). So the later rebinding of `termination_reason`
  (F03/F04) and `committed` does not reach them.

## 5. Identity hazards, re-checked at 5d295301

[read] These held before #527 and still hold:

1. **`retained_rank_info` and `certify_centered_factor` must not become bound methods of the
   fit object.**
   - A deferred `retained_rank_info` lives on the caller's model until
     `fit_ops._install_deferred_rank_info` runs it. A bound method would keep the whole fit alive
     (`_X_full`, both caches, every system); #492 bounded what this closure keeps alive.
   - Make both module-level functions bound with `functools.partial` over the values listed in
     §4, plus a small certifier object. The certifier holds only `dm`, the
     `_fisher_data_reuse` reference and the certification cell.
   - Keep `certify_centered_factor`'s side effect, which clears `_fisher_data_reuse`. A deferred
     call after the fit is latent (anatomy surprise 19).
2. **Never cache these in a hot-path local past a rebinding:** `_score_centre` (L03, L05, L06),
   `_observed_newton_active` and `_can_reuse_weighted_gram` (L16), and `_state_center` (S13).
   The binding points are in §4.
3. **`scop_merit_errors` is keyed by `id()`** (S22 2902; lookups at L09 3608).
   - The lookups use the very `_IRLSState` objects that `with_scop_merit` returns, so the
     search must receive them unwrapped and uncopied.
   - The dict is never pruned. It stays correct only while those states are alive. If one were
     freed, a later object could reuse its id and be read with the wrong allowance, silently.
   - So `scop_trial_cache`, `scop_committed` and the proposal must live at least as long as the
     search, as they do today.
4. **The factor-certificate cache tests identity:** `cached.system.data_gram is
   system.data_gram`, likewise `penalty`, `hessian`, `mean_x`, and `cached.system is system`
   (2662–2678).
   - The MSN right-hand-side adjustment builds a new system with `dataclasses.replace` (3366,
     3433). That keeps the arrays' identity, so the cache still hits.
   - `get_centered_system` returns the same cached object while `z_off` is unchanged.
   - Any copy, or a new `CenteredSystem` per call, turns a cache hit into a refactorisation. The
     bits stay the same, but the work counts change, and those are a gated count.

[read] **New, or newly important:**

5. **`SCOPRunCentring.bind((dm, family, link, *gms, *groups))`** (2271) compares its owners by
   identity.
   - The fit object must hand on the caller's `dm`, `family`, `link`, `gms` and `GroupSlice`
     objects themselves, never rebuilt ones.
   - Otherwise every fit clears the run's reduced design, supports and refusals. That costs time,
     and is bitwise neutral only where every carried refusal would recur.
6. **`_initial_data_key`** (2576–2586) holds `id(dm)`, `id(dm.execution_plan)`, `id(family)` and
   `id(link)`. **`fixed_owners`** (2508–2536) is a tuple of the same objects, and
   `_FisherDataReuse.take` keys on it. Neither may be wrapped.
7. **F04 judges by identity: `_judged_eta is eta_unclipped`** (4519).
   - The verdict on truncated directions is the fit's own only when the array the certificate
     read is the very `eta_unclipped` the loop retained.
   - So the split must pass `retained.eta_unclipped` into the certificate, and keep it as the
     post-loop `eta_unclipped`, with no copy.
   - Otherwise the record is published as history (`earlier=True`). That changes
     `truncated_directions` and can change the `mean_space_boundary` relabel.
   - F01 rebinds `eta_unclipped` for Gaussian/identity only, which never meets the Binomial/log
     certificate.
8. **The `scop_trial_cache` keys are floats.**
   - The search evaluates `alpha * shortened`, and the decision is relabelled to
     `decision.alpha * shortened` (3594, 3620).
   - The keys match because both are the same product of powers of two. Keep the multiplication,
     and its order, exactly.
9. **`_newton_rows_at` and `_trial_rows` cache by identity** (`state is committed`,
   `_trial_rows[0] is not state`). A copy breaks only the reuse: the rows come out the same, at
   one more log1mexp pass each.
10. **`_SCOPOrdinarySolve.centred_rows`** is `partial(_centred_system_matvec, _reduced_dm,
    system=reduced_centered)`. Each damped trial holds the iteration's reduced centred system
    until the next iteration replaces `scop_ordinary`, so the SCOP state must not keep older
    ones.

## 6. The strategy interface for S2–S4

[read] This is derived from §3–§5: the step blocks' writes and the fields every later block
reads. Signatures use today's names. "fit" is the S1 method object.

### 6.1 The fit object (S1)

- **Fields** are the 212 cross-block names. Block temporaries (`state`, `st`, `g`, `centered`,
  `z_off`, `result`) stay method locals, because different blocks reuse the names with different
  meanings.
- **The setup order is fixed by late binding:**
  - bind `state_center` (S03, then S13) before the first evaluation (S16);
  - bind `has_scop` (S13) before any evaluation;
  - bind `mode_bar` and the certificate cells (S21) before the loop.
- **Methods.** `run()` is the loop plus finalisation. The closures that read only fixed setup
  become methods (`PenaltyOps`, the emitters). The per-iteration closures stay local to the
  iteration method.

### 6.2 Collaborators

```python
class PenaltyOps:                      # S06; fixed per fit
    def matvec(self, beta, *, magnitude=False) -> NDArray
    def rounding(self) -> int
    def curvature(self) -> NDArray     # diag(S), cached on first use
    def quadratic(self, beta) -> float
    def row_set_curvature(self, sets) -> NDArray   # cached (S08)

class StateEvaluator:                  # S09-S11; built after has_scop and state_center (S13)
    def evaluate(self, beta, intercept, *, phase, iteration, alpha=None, deviance=None,
                 eta_unclipped=None, enclosing_proposal_state_id=None, emit_trace=True,
                 centred_intercept=None) -> _IRLSState
    def emit_evaluation(self, state, *, phase, iteration, alpha=None,
                        enclosing_proposal_state_id=None, deviance_reused=False) -> None

class CenteredSystemBuilder:           # S18-S20; QR, dense, QP and the centred finaliser
    def system(self, W, z_off) -> CenteredSystem                       # get_centered_system
    certifier: CenteredFactorCertifier                                  # holds dm, fisher-reuse ref, cell
    def disable_weighted_gram_reuse(self) -> None                       # the Gamma/log switch
    def carry_refusals(self) -> None                                    # into SCOPRunCentring

class CenteredFactorCertifier:         # what retained_rank_info may capture (hazard 1)
    def __call__(self, system, W, *, response=None) -> _CenteredFactorCertification

class CurvaturePolicy:                 # S15, S15b, L02's choice, L16's switch, the latch
    observed_active: bool              # read at call time by the builder and L02
    stagnation_newton_available: bool
    def working_rows(self, committed) -> CoefficientWorkingRows        # prefer_observed=observed_active
    def observed_rows_invalid(self, candidate) -> bool
    def switch_to_newton(self, builder, profile) -> None               # sets latch, disables Gram reuse
```

### 6.3 The step strategy

The fit chooses one strategy at setup, from `_use_qr`, `_has_scop`, `_use_structured` and
`has_constraints`. The S0 refusal guarantees SCOP and the QP never meet.

```python
@dataclass
class WorkingRows:                     # L02's output: every step's input
    rows: CoefficientWorkingRows       # rows.curvature_source: "fisher" | "observed"
    W: NDArray                         # rows.weights, or the MSN observed curvature
    z: NDArray                         # rows.response, or committed.eta_unclipped under MSN
    newton_score: NDArray | None       # MSN only
    newton_committed_rows: NDArray | None
    positive_w_min: float; positive_w_max: float; w_ratio: float   # Fisher statistics (F02, L20)

@dataclass
class Proposal:                        # L03-L07's output
    beta: NDArray
    intercept: float
    centred_intercept: float | None    # QR, dense, structured with a state centre
    cond_est: float
    used_svd: bool
    rank_truncated: bool | None
    used_rank_certification: bool      # dense, SCOP
    held_rank: RankDecomposition | None          # QR, dense; the certificate reads it
    score_centre: tuple | None                   # QR, dense, structured, under mode_score; kept
                                                 # across iterations when a step sets none

@dataclass
class SearchOutcome:                   # L09 or L10-L15
    decision: _IRLSStepDecision
    proposal_state: _IRLSState
    retained: _IRLSState               # the same object the search returned (hazards 3, 7)
    qp_certified: bool                 # retained_qp_converged; True off the QP

class Step(Protocol):
    backend: str                       # "qr" | "gram" | "structured" (F12); SCOP and QP are "gram"
    def initial_state(self, committed: _IRLSState) -> _IRLSState
        # S22: SCOP builds its latent bundle and merit (objective scale 0); the others return it
    def propose(self, work: WorkingRows, committed: _IRLSState, iteration: int) -> Proposal
        # L03 | L04 | L05 | L06 (+ L07 for the QP); L08's SVD warning stays shared
    def search(self, committed, proposal, work, iteration) -> SearchOutcome
        # SCOP: its own (L09); the others: the shared TrialSearch with the step's validity hook
    def latent_change(self, outcome) -> float | None   # SCOP latent criterion (L16), else None
    def damped(self, outcome) -> bool                  # SCOP: a damped step never stops (L16)
    def certificate_inputs(self, outcome) -> tuple[RankDecomposition | None, NDArray | None, tuple | None]
        # (held_rank, active_rows, scop_groups) for true_mode_residual (L17)
    def commit(self, outcome) -> None                  # L21: SCOP rebinds scop_committed
    last_working_system: CenteredSystem | StructuredSystem | None   # _return_working_system (F08)
    def terminal(self, mu, eta, eta_unclipped) -> None # F05: SCOP block_curvature; the others none
    finaliser: CentredFinaliser | StructuredFinaliser  # F08-F11
    def postfit(self, result, finals) -> dict | None   # F14: SCOP inference, phi, scop_converged
```

The QP is a sub-mode of `DenseStep`. It holds `A_all`, `b_all`, `abs_A_all`, the active set,
the committed active set, the #472 retry latch and the reported-once flag. `FeasibilityContract`
reads them and owns the checks the loop makes whatever the step:
- the committed and proposal feasibility (L11) and the trial predicate;
- the infeasible-committed override (L15);
- the veto (L18) and the final refusal (F03).

### 6.4 The mean-space overlay (#425 guard, #437 Newton and certificate)

```python
class MeanSpaceOverlay:
    available: bool                     # S17: Binomial/log, not QR, SCOP or _return_working_system
    engaged: bool                       # _mean_space_newton; L17 sets it
    def rows(self, committed) -> tuple[NDArray, NDArray, NDArray, NDArray] | None   # L02
    def centred_rhs(self, centered, newton_score) -> CenteredSystem                 # L06, L07
    def merit_hooks(self, committed, newton_score) -> tuple[Callable, Callable]     # L12
    def halving_budget(...) / first_halving(...)                                     # L14
    def judge(self, verdict, retained, committed, certificate_inputs, iteration) -> Verdict  # L17
    def final(self, eta_unclipped, converged, termination_reason) -> FinalVerdict     # F04
    def profile(self, profile) -> None                                                # F07
```

- The **Newton half** acts on the dense, QP and structured steps: rows, right-hand sides,
  increments and merit.
- The **certificate half** (`judge`, `final`) acts on every step, through
  `certificate_inputs`.
- The mean-space and link-domain predicates (L13) are #425's guard, not #437's. They go into
  `TrialSearch.validity`, so the second PR can retire #437 without touching them.

### 6.5 Convergence, the recorder and the finalisers

```python
class Convergence:                     # L16; its order is the anatomy's precedence
    def judge(self, iteration, committed, retained, previous, dev, objective_prev,
              outcome, proposal, step, policy, builder) -> Verdict
    # Verdict: converged, score_stagnated, value, label, tolerance, coef_change, dev_rel_change
    # It calls policy.switch_to_newton(builder, profile) itself, never by rebinding a name.

class Recorder:                        # S09/S11 emitters live in StateEvaluator; L19, L20, F07, F12
    def step_decision(...) -> None; def iteration(...) -> None; def finish(...) -> None
```

- **The loop body, in this order:** `work` (L02), `step.propose`, the L08 warning,
  `step.search`, `convergence.judge`, `overlay.judge`, `feasibility.veto`, the reason chain,
  `recorder`, then `deviance_relative_change` and the L21 control flow.
- **After the loop:** F01 rebinds eta, mu and dev; then F02, F03, F04 and the truncated relabel,
  `step.terminal` (F05), the export rows (F06), `recorder.finish` (F07), the finaliser (F08–F11)
  and F12–F14.
- **The two finalisers:**
  - **Centred**, used by QR, dense, QP and SCOP: the export system through
    `CenteredSystemBuilder`, `cache_out`, and the slope pseudo-inverse with `RankInfo` (deferred
    through the partial in hazard 1).
  - **Structured**: the profiled block factor and its EDF.

### 6.6 Where the map contradicts the design

1. **SCOP has one form.** The design's "SCOP block-coordinate" harness mode, its "`_scop_joint` is
   a per-fit parameter" bullet and the plan to delete block-coordinate SCOP in the split were
   all overtaken by #527. Struck in the design copy.
2. **SCOP is a client of the curvature policy.** The design (and the anatomy, and the function's
   own docstring) say SCOP keeps Fisher. Since #527 SCOP takes observed rows and the observed-row
   predicate, so `CurvaturePolicy` serves `SCOPStep` as well.
3. **The overlay is wider than the design says.** It changes the QP step too (L07's right-hand
   side), and its certificate half runs on QR and SCOP. That is the anatomy's correction, and it
   is still true.
4. **Convergence is not pure.**
   - It switches curvature (Gamma/log, L16) and engages Newton (Binomial/log, L17).
   - Since #527 it also asks the step whether the step was damped (`damped`) and for its latent
     change.
   - The design's "convergence covers the deviance, coefficient and mode-score criteria and the
     score centre" needs those three hooks.
5. **The recorder is bigger, and keeps one write out.** Instrumentation is 494 lines (12%): trace,
   diagnostics, debug rows, profile, timers and logging. `deviance_relative_change` stays in the
   loop, because F02 reads it.
6. **The line search is split.** SCOP's search (L09) is its own: a latent interpolation with an
   `id()`-keyed merit allowance and no domain guard. Only dense, QP, QR and structured share
   `TrialSearch`. The design's single "line search" module is two.

## 7. Is the length lines-of-code abuse?

**No, in the sense Max means: little padding and little copy-paste.**

- Comments and docstrings are 775 of the 3,999 lines (19%) and blank lines 129. The comments
  state rounding bounds and the why of numerical choices; they are not filler.
- The exact-type gate is copied once (S18, anatomy surprise 21), and the two Newton switches are
  one method written twice (anatomy §2). Little else repeats.

**It is long because it multiplexes.**
- Four step methods and one overlay share one scope and 212 cross-block names. Each mode's code
  is spread through setup, step, search, convergence and finalisation behind 257 `if`s, nested up
  to 8 deep.
- SCOP's flag `_has_scop` alone is read in 16 of the 61 blocks, plus `evaluate_state`'s capture.
- So every change has to be read against every mode. That is what the split fixes.

[AST] Exclusive line counts. Each line takes its block's mode, unless the innermost `if` on a
mode flag, an instrumentation statement or a mode's nested `def` decides otherwise. Every line is
counted once.

| What | Lines | Code | Comment and docstring | Share |
|---|---|---|---|---|
| SCOP (S16, L04, L09, F05, and SCOP branches in S22, L16 and F14) | 685 | 619 | 53 | 17% |
| Binomial/log mean space (#425 guard, #437 Newton, certificate S08) | 577 | 422 | 136 | 14% |
| Instrumentation (trace, diagnostics, debug rows, profile, timers, logging) | 494 | 468 | 13 | 12% |
| Shared glue (setup, loop control, state, penalty operators) | 475 | 308 | 132 | 12% |
| Structured (S03, L05, F10) | 288 | 232 | 54 | 7% |
| Centred-system builder and caches (S18–S20) | 236 | 211 | 21 | 6% |
| Shared finalisation (F01, F02, F06, F08, F09, F13, F14 core) | 233 | 162 | 58 | 6% |
| Convergence (S07, L16, L18 reasons) | 220 | 154 | 61 | 6% |
| Dense step and finaliser (L06, F11) | 206 | 160 | 39 | 5% |
| Signature and docstring (S01) | 177 | 50 | 123 | 4% |
| Shared line search (L10, L11, L13–L15) | 148 | 122 | 19 | 4% |
| QP (S12, L07, F03, and constraint branches) | 127 | 102 | 21 | 3% |
| Curvature policy (S15, S15b, the L16 switch) | 86 | 39 | 44 | 2% |
| QR (L03, S17's QR part) | 47 | 46 | 1 | 1% |
| **Total** | **3,999** | **3,095** | **775** | |

Caveats on the attribution:
- A test that names two modes goes to the first match in the order SCOP, structured, QR, QP,
  MSN, Newton. For example, L05's truncated-increment branch keys on both the sz factor and
  `newton_score`, so it counts as MSN.
- Logging lines count as instrumentation, even inside a mode block.

**#437's contribution.** 574 blame lines (14% of the function, second only to #156). Of these,
298 are the certificate (S08). The rest are the Newton hooks in L02, L05, L06, L07, L12, L14 and
L17, and F04/F07. That is the part the design's second PR replaces with the constrained-QP
method. The anatomy's §5.4 counts what that removes, about 419 lines in the function.

## 8. Known behaviour change, out of scope: SCOP warm starts enter raw

Found in #527's review (round 6, Low, not reproduced, no worse than master). A warm-started SCOP
inner fit evaluates its entry state from `eta = X β + intercept` formed raw. These are the REML
fits after the bootstrap, and the certification retries.

- **Where in the code:**
  - S22 calls `evaluate_state(..., centred_intercept=centred_start)` (2802).
  - `centred_start` needs `_state_center`, and SCOP has none (S13).
  - So iteration 1's `committed_level` comes from the raw intercept (L04, 3175–3184), rounded at
    `u |m' β|`, about 0.06 at a 1.5e15 column translation.
  - A damped first step can then be refused, ending the inner fit `step_rejected`.
- **The split keeps this bit for bit.**
- **Where the fix would go:** `SCOPStep.initial_state` (S22's SCOP branch). Today the first
  `_SCOPTrialState` is built there with `level=None`.
  - **Option one:** carry `(level, centre)` across fits, on `SCOPRunCentring` or through
    `_centred_init` as the ordinary route does. Seed `scop_committed.level/centre` from it, and
    form the entry eta from the carried level (`_level_about`).
  - **Option two:** form the entry eta centred.
- Either option changes the entry state of warm-started SCOP fits, so it is a behaviour change
  for its own PR, after the split.

## 9. What the S0 harness pins, and what it does not reach

[run] It reaches 94.7% of `_fit_irls_direct_once`'s statements on master (1,265 of 1,336) and
100% of `fit_irls_direct`.

Never reached, with the master line numbers (subtract nothing; the branch is 25 lines later):
- the structured route's "no compact `reml_penalties`" refusal and fallback (1359–1365);
- the raw (uncentred) `mode_residual` branch (1550–1552);
- these parts of the true-score certificate:
  - the non-finite and underflow refusals;
  - the NNLS iteration-limit fallback;
  - the weak-block decrement noise (`likelihood_noise`, `weak_penalty_block`);
  - joint discarded directions across SCOP groups (1902–1913);
- the cold raw-spline policy counter (2524) and an initial-data reuse hit (2575);
- certification clearing the Fisher-data reuse (2633);
- a non-finite observed entry state (2999);
- the Levenberg shift on a refused structured observed iterate (3248–3251);
- the truncated-sz increment without a state centre (3291, 3315–3325);
- the consecutive-SVD warning (3464–3468);
- a mean-space trial refusal (3768);
- non-finite deviance (4036–4040, 4166, 4293);
- the in-loop KKT-incomplete reason (4162) and the final KKT refusal (4462–4464);
- `SeparationError` (4445);
- `_compute_reml_geometry=False` on the structured route (4777–4779). No caller combines them:
  only SCOP candidates pass it, and SCOP is never structured;
- the unpenalised data-rank certificate (4905);
- the SCOP return without `return_xtwx` (5144).

The Levenberg branch has no fit-level fixture anywhere in the suite; the unit tests exercise
the shifted factor directly.

**What the harness reaches** [run], over 249 cases with calls and 4,386 calls:
- the gram backend in 160 cases, QR in 25 and structured in 64; SCOP in 48;
- linear constraints in 25 cases, with 152 QP solves, 124 of them with active rows;
- mean-space Newton in 8 cases (29 iterations), the Gamma/log switch in 1 and observed Newton in
  35;
- score stagnation in 4 cases and the mean-space boundary in 8; 12 traced cases.

**The S0 gate** [run], same machine, 10 cores:
- master 5d295301 against master: 252 of 252 cases identical;
- master against the branch (the refusal and the harness): 252 of 252.

**The harness is not blind.** Against the same master run, three mutations of `irls_direct.py`
each make `compare` exit 1:

| Mutation | Cases that differ | First difference |
|---|---|---|
| `penalty_quadratic` summed in reverse | 11 | a convergence value one ulp apart |
| `max_halving` 20 → 19 | 23 | `trials_attempted` 21 → 20 |
| `score_stagnated` reported as `max_iter` | 6 | an iteration's `termination_reason` |

To run it against the stored master baseline (`audit042/irls_split/s0_out/master_final`):

```
NUMBA_CACHE_DIR=<s0_out>/numba_cache benchboard heavy <label> --cores 10 --wait-max 240 -- \
  timeout 1200 <tree>/.venv/bin/python <tree>/benchmarks/irls_direct_characterisation.py \
  run --out <dir> --jobs 10
python3 <tree>/benchmarks/irls_direct_characterisation.py compare <s0_out>/master_final <dir>
```
