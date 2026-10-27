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

### 6.7 As built (S3a to S5)

The package `superglm/solvers/irls_direct/`:

| Module | Holds |
|---|---|
| `__init__` | The solver's docstring; re-exports the 17 names other modules and tests import |
| `_fit` | `fit_irls_direct`, `_IRLSDirectFit` and its fields, the centred system's factor certificate (`_certify_centered_factor`) and the last Gram decomposition (`_authoritative_gram_rank`), the module helpers the fit reads, and (S3d) `_IRLSDirectCurvaturePolicy` |
| `_finalise` | (S5) the centred finaliser `_finalise_centred` (with `_final_centred_system`), its `_feature_edf` and `_retained_rank_info`, and `_IRLSDirectFinalGeometry`, what every finaliser returns |
| `_dense` | (S3d) `_IRLSDirectDenseStep` |
| `_qp` | (S3d) `_IRLSDirectQPStep` and `_solve_constrained_qp_with_cold_retry`; (S4) the feasibility contract's tolerance `_QP_FEASIBILITY_TOL`, beside the step's `feasible` |
| `_convergence` | (S4) `_IRLSDirectConvergence`, the stop rule |
| `_search` | (S4) `_IRLSDirectTrialSearch`, the shared line search |
| `_mean_space` | (S3d) `_IRLSDirectMeanSpaceOverlay`; the certificate's helpers `_SUBNORMAL_SPACING`, `_underflow_allowance`, `_PENALTY_HEADROOM_EXPONENT`, `_even_exponent` and `_column_shift` |
| `_recorder` | `_IRLSDirectRecorder` |
| `_step` | `_IRLSDirectStep`, the interface (a `Protocol`; since S3c its optional hooks default to `None` and every step subclasses it); (S3c) `_note_svd_fallback`, L08 |
| `_qr` | `_IRLSDirectQRStep`, `_sqrt_penalty_augmented`, `_centred_rows` |
| `_structured` | (S3b) `_IRLSDirectStructuredStep`; `StructuredSolverError` (its `__module__` still the package) and `_structured_solver_errors`; the Levenberg shifts (`_LEVENBERG_SHIFTS`, `_levenberg_shifted_operator`, `_levenberg_shifted_leaf_operator`, `_build_iterate_factor`); `_structured_score_centre` |
| `_centred` | `_system_offset_mean`, read by QR, the dense step and the centred finaliser |
| `_scop` | (S3c) `_IRLSDirectSCOPStep`; `SCOPRunCentring` with `_seed_refusals` and `_carry_refusals` (which `_fit`'s centred system also calls); the latent state and trial records (`_SCOPGroupState`, `_SCOPTrialState`, `_SCOPOrdinarySolve`), `_evaluate_scop_trial`, `_level_about`, `_LATENT_MAX_STEP`, `_latent_step_depth`, `_centred_system_matvec`, `_lone_scop_support` |

A strategy module never imports `_fit`; `_fit` imports the strategies. A helper that a
strategy and the fit both read goes to a leaf module such as `_centred`.

The interface as it stands:

```python
class _IRLSDirectStep(Protocol):      # (S3c) every step subclasses it; each hook defaults to None
    backend: str                       # what the fit publishes ("qr", "gram", "structured")
    finalise: Callable[[_IRLSDirectFit], _IRLSDirectFinalGeometry]
                                       # (S5) no default: every step names its finaliser
    data_rank: Callable[[CenteredSystem, NDArray], RankDecomposition] | None
                                       # read by the centred finaliser
    # (S3c) the SCOP step's hooks; None on every other step
    initial_state: Callable[[_IRLSDirectFit, _IRLSState], _IRLSState] | None   # S22
    search: Callable[[_IRLSDirectFit, int], None] | None    # L09; None: the shared search
    latent_change: Callable[[], float] | None               # L16, the coefficient criterion
    damped: Callable[[_IRLSDirectFit], bool] | None         # L16, a damped step never stops
    terminal: Callable[[_IRLSDirectFit], None] | None       # F05, after the export rows (S5)
    postfit: Callable[[_IRLSDirectFit, PIRLSResult, _IRLSDirectFinalGeometry], dict | None] | None
                                                            # F14, F09's eta
    def propose(self, fit, it: int) -> None
    def certificate_inputs(self) -> tuple   # (S3d) L17's inputs; default (None, None, None, None)
```

Where §6.3 changed, and why:

1. **The fit is an argument, never a field.** A step that kept the fit would be a cycle
   only the collector frees, the lifetime regression #492 removed.
2. **No `WorkingRows`, `Proposal` or `SearchOutcome` objects.** The step reads `W`, `z`
   and `newton_score` from the fit and writes the proposal to the fit's slots. An object
   per iteration would be a new allocation and ten reference copies, and the slots
   already are that record.
3. **`finaliser` became `data_rank`.** QR's only finalisation of its own is the retained
   fit's data rank, from its dense rows; the rest is the centred finaliser. A step's own
   data rank holds the step, so the fit computes it at once and never defers it.
4. **QR needs none of `search`, `latent_change`, `damped`, `certificate_inputs`,
   `commit`, `terminal` or `postfit`.** It takes the shared search, and its held rank
   reaches the certificate through the fit's `_held_rank` cell, as the dense step's does.
   Each arrives with the strategy that needs it. The SCOP-or-shared search choice stays
   in `run()` until SCOP is a strategy.
5. **The Gram-path routes are one step for now.** `_IRLSDirectGramStep.propose` is the
   old `_step`'s Gram branch, which dispatches to `_step_scop`, `_step_structured`,
   `_step_dense` and `_step_qp` on the fit's flags. `_publish` still reads "structured"
   from `_use_structured`.
6. **L08, the consecutive-SVD warning, is not shared.** It is the Gram-path step's; QR
   never ran it.
7. **The step is chosen where S17 chose QR, in `_setup_systems`,** so the
   discretized-design warning keeps its place. `_use_qr` remains a local there, for the
   centred builder's gates.

Where S3b (the structured step) changed it:

8. **`finalise` is the structured finaliser.** F08's and F10's structured branches and
   F09's structured `cache_out` entries are one method of the step. It writes what
   publication reads of the geometry: `XtWX_beta`, `XtWX_S_inv_beta`, `log_det_H`,
   `reml_hessian_rank`, `p_eff` and `structured_factor`, and `None` for
   `reml_geometry_summary`, `retained_reml_decomposition` and `rank_info`. A step with
   `finalise = None` keeps the centred finaliser, the fit's `_final_system` and
   `_final_geometry`, which now hold the centred route alone. The structured finaliser
   builds no `RankInfo`, so `data_rank` is `None` there only to satisfy the interface.
   Publication reads the centred moments (`XtWX`, `XtW1`, `sum_W`, `centered_final`) in
   SCOP's post-fit inference alone, their one reader.
9. **The structured step is chosen in `_setup_design`**, where `resolve_structured_backend`
   chooses the route, because its layout fixes the state's centre before the first state
   is evaluated. Its constructor applies the fs BLAS cap (`keep_narrow_cap`) where the fit
   did; `state_centre` forms `c0`. QR and Gram are still chosen in `_setup_systems`.
10. **The step keeps its own last system**, which the fREML performance iteration
    publishes. The fit's `_structured_group_index`, `_structured_layout`,
    `_last_working_structured`, `_final_penalized_operator`, `structured_final` and
    `mean_x` are gone. The step's write of `_final_penalized_operator` was dead: the
    finaliser always rebuilt the operator from the final system.
11. **No search hook and no L08.** The structured route takes the shared search, which
    never read the route. L08 is the Gram step's alone: on the structured route
    `_used_svd` is always False, so it only ever reset the counter to 0.
12. **`_use_structured` is a setup field now:** the gate of the Gram state centre
    (`_setup_routes`) and the centred builder's four gates (`_setup_systems`) read it.
    Every branch on it after setup is gone, `_publish`'s included (`step.backend`).
13. **For the queued "reuse the last decomposition of the same Hessian" patches**
    (centred route): the structured analogue fits inside the step. Under
    `_return_working_system` the final system is the step's last system, and its
    unshifted factor is the one the last iteration took; keeping that factor and its shift
    on the step would let `finalise` reuse it.

Where S3c (the SCOP step) changed it:

14. **The SCOP step is chosen at the end of `_setup_routes`,** where `_has_scop` is
    decided.  Its constructor is the old `_setup_scop`: it evaluates the start
    (`scop_initialization`), so it follows every field `evaluate_state` reads.  It writes
    two fit fields: the first latent coordinates' mapped coefficients into `beta`, and
    `_scop_run`, whose full-design refusals the centred system seeds and carries.  The
    REML curvature class (`classify_scop_reml_curvature`) is still computed where it was,
    before the refusals, and handed to the constructor.
15. **The hooks.**
    - The interface (`_step`) gives each optional hook a `None` default, and every step
      subclasses it.  The `data_rank = finalise = None` lines are gone from the steps.
    - The hooks are `initial_state(fit, committed)` (S22), `search(fit, it)` (L09),
      `latent_change()` and `damped(fit)` (L16), `latent_groups()` (L17's SCOP input),
      `terminal(fit)` (F05) and `postfit(fit, result)` (F14).
    - `postfit` also writes F09's `cache_out["eta_unclipped"]`.  Nothing writes
      `cache_out` between there and `postfit`, so the dict's order is unchanged.
    - `search` writes what the shared search writes: `retained` with its `eta`, `mu` and
      deviance, `beta`, `intercept`, `decision`, `n_halvings`, `step_rejected` and
      `proposal_state`.
16. **No `commit` hook.**
    - L21 commits the retained state or ends the loop.  So the step's committed latent
      state at `propose` is the one its last search retained.
    - `initial_state` seeds that state with the fit-entry bundle.
    - `run()` reads no `_has_scop`, and the search choice is one local `None` test.
17. **Convergence goes through hooks, not `_has_scop`.**  This keeps S4 simplest:
    `_converge` reads no route flag.
    - `latent_change` joins the coefficient criterion, and `damped` vetoes a stop under
      either criterion.
    - The fixed-point test's `not _has_scop` was dead, because SCOP never sets
      `_score_centre`, and is deleted.
    - Both searches write `proposal_state`, so the `proposal` field and `_converge`'s
      choice between the two are gone.
18. **The certificate reads `latent_groups()`.**
    - It returns `(state, spec)` pairs.  `true_mode_residual` reads the spec's `group`,
      `S_scop`, `B_scop`, `bin_idx` and `reparam`, the objects the step's latent-state
      dict holds, in place of the fit's `_scop_state`.
    - `_judge_mean_space` reads the hook only where the certificate runs.
    - `_scop_columns` stays a fit field for S3d's `certificate_inputs`.
19. **Fields that left the fit (25).**
    - Setup: `_S_reduced`, `_reduced_to_full`, `_reduced_dm`, `_reduced_tabmat_state`,
      `_scop_specs`, `_scop_order`, `_scop_newton`, `_expose_exact_support_state`,
      `_scop_state` and `_scop_curvature`.
    - The merit: `scop_outer_penalty`, `scop_outer_penalty_abs`, `scop_merit_errors`,
      `merit_gamma`, `gaussian_response_roundoff` and `nb2_unit_roundoff`.
    - The iteration: `scop_committed`, `scop_proposal_eta_unclipped` (and its reset in
      `_working`), `scop_ordinary`, `centred_level`, `scop_results`, `proposal_scop`,
      `retained_scop` and `proposal`.
    - They are the step's slots now.  `_has_scop`, `_scop_columns` and `_scop_run` stay.
20. **`_has_scop` after S3c** (18 mentions, from 33).  It is a setup field:
    - `_setup_routes` reads it for the state centre's gate, the refusal, the stagnation
      Newton's exclusion and the step choice;
    - `_setup_systems` reads it for QR, the mean-space Newton's availability, the Gram
      caches, `fixed_owners` and the initial-data reuse;
    - `evaluate_state` reads it, because a SCOP state's merit is the step's;
    - so does the debug row.
    No branch after setup reads it.
21. **L08 is `_note_svd_fallback(fit)`,** which the Gram step and the SCOP step call
    after their solve, as each ran it before.
22. **The identity hazards hold.**
    - `_merit_errors` is still keyed by `id()` and still never pruned.
    - The trial cache is still a local of `search`, keyed on `alpha * shortened`.
    - `bind` gets the fit's own owners.
    - `search` writes `retained.eta_unclipped` itself into the fit (F04).
    - The proposal's state outlives the search through `proposal_state`, and the
      committed and retained trial states through the step's slots.

Where S3d (dense, the QP, the mean-space overlay and the curvature policy) changed it:

23. **Dense and the QP are two steps, not one step with a QP part.**
    - L06 and L07 share only the centred system and the overlay's right-hand side
      (§3). Their writes are disjoint. L06 writes the held rank, the rank
      certification, the score centre and the centred intercept. L07 writes the
      active set, the QP certificate and the #472 retry and report latches.
    - A constrained fit never has a state centre (S13), a score centre or
      observed curvature (S15).
    - One step with a QP part would keep a per-iteration branch on "constrained"
      inside `propose`, which is the multiplexing the split removes. §6.3's "the QP
      is a sub-mode of `DenseStep`" was written before S3a, where every route
      became its own step class.
    - **Where each is chosen.** The QP step is chosen in `_setup_routes`, where the
      constraints are found, and it builds `A`, `b` and `|A|` once. The dense step
      is chosen in `_setup_systems`, beside QR.
    - **What the QP step holds:**
      - `A_all`, `b_all`, `abs_A_all`;
      - `active_set`, formerly the fit's `prev_active_set`;
      - `committed_active_set`, L01's copy, now taken at the start of `propose`;
      - `proposal_converged` and `retained_converged`;
      - `_degeneracy_retry_failed` and `_reported_nonconvergence`.
    - **The fit's `qp` field** is the QP step or `None`. After setup it replaces
      every read of `has_constraints`, which is now a local of `_setup_routes`
      and `_setup_systems`.
    - **What each step reads.** The dense step reads `decompose_gram_if_authoritative`
      in `_dense`. The QP step reads `solve_constrained_qp` in `_qp`, through
      `_solve_constrained_qp_with_cold_retry`.
24. **The feasibility contract stays in the fit and reads `qp`.**
    - L11 (`_search`): the feasibility of the committed state and of the
      proposal, and the trial predicate, on `qp.A_all`, `qp.b_all` and
      `qp.abs_A_all`.
    - L15: the infeasible-committed override. It also writes
      `qp.retained_converged` and, on a rejected step, restores
      `qp.active_set = qp.committed_active_set`.
    - L18 (`_terminate`): the feasibility of the retained coefficients, and
      `qp.retained_converged`.
    - F03 (`_finish`): the final refusal, the same two tests.
    - `_QP_FEASIBILITY_TOL` stays in `_fit` with the contract.
    - **Dead code removed:**
      - L01's per-iteration resets of both certificates. L15 always writes them
        before L18 or F03 reads them.
      - The three "construction invariant" checks on `A_all`.
      - The fixed-point stop's `has_constraints` test. A QP fit never sets
        `_score_centre`, so the mode-score branch never runs on it; this is
        S3c's deletion of `not _has_scop`, made again.
25. **`_IRLSDirectCurvaturePolicy`, in `_fit` (the fit's `curvature`).** It owns
    Fisher versus observed curvature.
    - It is constructed in `_setup_routes`, at S15's place. S15's comment moved
      into its constructor.
    - It holds `family`, `link`, `y`, `weights` and the latch list, and no
      reference to the fit.
    - It lives in `_fit`, beside the centred system that reads it. The two module
      names it calls, `coefficient_working_rows` and `_has_constant_irls_weights`,
      are what nine tests patch at `_fit`. Their intent covers L02 together with
      the mode score's and the export rows' calls.
    - `_has_constant_irls_weights` is now called once, not twice.

    | Member | Set by | Read by |
    |---|---|---|
    | `observed` | constructor (S15, then the latch); `switch_to_newton` | `working_rows` (L02); `_search` (L11's trial predicate); the SCOP step's `search`; `get_centered_system` (the Fisher-data reuse) |
    | `stagnation_newton_available` | constructor | `_converge` (the Gamma/log switch's condition) |
    | `weighted_gram_reuse` (formerly the fit's `_can_reuse_weighted_gram`) | constructor (the S18 gate: constant weights, not SCOP, not structured, not observed); `switch_to_newton` | `_setup_systems` (the raw-spline deferral's `constant_weights`); `get_centered_system` (the constant-weight cache and the Fisher-data reuse) |
    | `working_rows(mu, eta)` | — | `_working` (L02) |
    | `observed_rows_invalid(candidate)` | — | `_search` and the SCOP `search`, which pass it while `observed` |
    | `switch_to_newton(recorder)` | — | `_converge` (L16): sets `observed`, turns `weighted_gram_reuse` off, sets the latch and counts `irls_fisher_newton_switches` |

    - Every reader reads the policy's field where it uses it. The policy object
      itself is never rebound, so the late-binding hazard (§5, item 2) has no
      name left to cache.
    - `disable_weighted_gram_reuse()` (§6.2) is part of `switch_to_newton`. The
      flag is the policy's own, and it changes only when the curvature does.
      Two separate calls could leave one done without the other.
26. **`_IRLSDirectMeanSpaceOverlay`, in `_mean_space` (the fit's `mean_space`).**
    - It holds all of #437's Newton steps, and the true-score certificate of #437
      and #454.
    - The fit holds one when `mean_space_violation(family, link)` is not `None`
      (binomial/log), else `None`, and every consult tests that.
    - It is built in `_setup_systems`, at S17's place: `available = not (_use_qr
      or _has_scop or _return_working_system)`, with `start_lowered` taken from
      `_setup_design` (S04).
    - It keeps no reference to the fit.
    - **Its surface, every call site:**

    | Method | Called from | Reads from the fit | Writes to the fit | Its own state |
    |---|---|---|---|---|
    | `newton_rows(fit, committed)` | `_working` (L02), while `engaged` | `family`, `link`, `y`, `weights`, `recorder` | none: returns `(W, newton_score, z)`, or `None` at the boundary or without rows, and `_working` stores them | writes `_committed_rows`; counts `irls_mean_space_newton_iters` |
    | `centred_rhs(centered, newton_score, dm)` (static) | `_IRLSDirectDenseStep.propose` (L06), `_IRLSDirectQPStep.propose` (L07), when `newton_score` is set | none | none | none |
    | `newton_iterate(fit, rank, centered, newton_score)` (static) | `_IRLSDirectDenseStep.propose` (L06) | `committed`, `dm`, `penalty_matvec` | none | none |
    | `line_search(fit, committed, proposal, newton_score)` | `_search` (L12, L14), once per iteration | `weights`, `family`, `link`, `max_halving`, `y` | none: returns `(first_halving, lowered_budget, deviance_delta, merit_roundoff)` | reads `start_lowered` and `_committed_rows` |
    | `judge(fit)` | `run()`, after `_converge` and before `_terminate` (L17) | `_stagnation_window`, `converged_this_iter`, `score_stagnated`, `family`, `link`, `retained`, `committed`, `weights`, `dev`, `step_rejected`, `mode_bar`, `step.certificate_inputs()` | `converged_this_iter`, `score_stagnated`; under Newton steps also `convergence_label`, `convergence_value` and `convergence_tolerance` | reads `available`, `start_lowered`, `engaged` and `_true_ratios`; writes `engaged`, `true_stop`, `true_confirmed`, `true_stop_ratio` and `_true_ratios` |
    | `certify(fit, beta, intercept, eta, inputs)` | `judge` | `dm`, `lambda2`, `p`, `n`, `y`, `weights`, `offset`, `mode_bar`, `_penalty_curvature`, `penalty_matvec`, `penalty_rounding`, `penalty_curvature`, `_laplace_excluded`, `recorder` | `_penalty_curvature` (filled on first use) | writes `last_true_residual`, `_judged_truncated`, `_last_truncated` and `_row_set_curvature` |
    | `final(fit, eta_unclipped, converged, termination_reason)` | `_finish`, after F03 (F04) | `family`, `link`, `weights` | none: returns `(converged, termination_reason)` | reads `_judged_truncated` (by identity against `eta_unclipped`) and `_last_truncated`; writes `boundary_rows`, `published_truncated` and `truncated_boundary` |

    - **Other readers of its fields:**
      - the recorder's `profile_fit` (F07): `boundary_rows`, `true_stop`,
        `last_true_residual`, `true_stop_ratio`;
      - `_publish` (F14): `boundary_rows`, `true_confirmed`,
        `published_truncated`, `truncated_boundary`, with `0`, `False`, `()` and
        `False` when the fit holds no overlay.
    - **`newton_score` stays a fit field.** It is a working-row output, like `W`
      and `z`: `_working` sets it from `newton_rows`. Its readers:
      - the dense and QP steps, which call the overlay;
      - `_search`, which passes it to `line_search`;
      - the recorder's working-curvature label;
      - the structured step, which consults the Newton half at four sites of
        its own: `Wz = W z_off + newton_score`, the truncated-increment condition
        `or newton_score is not None`, the `irls_truncated_newton_increments`
        count, and `residual_rows = newton_score`. That increment is the
        structured step's own: `sz` takes it too, Pes & Rodriguez.
    - **What each step answers to `certificate_inputs()`:**
      - QR and dense: their decomposition (`_held_rank`, now a slot of each
        step);
      - QP: `A_all[active_set]`;
      - SCOP: its columns (`_columns`, formerly the fit's `_scop_columns`) and
        the retained latent groups with their specs. This replaces S3c's
        `latent_groups` hook.
      - structured: the default, all `None`.
    - **Not in the overlay:**
      - L13's mean-space and link-domain trial predicates (#425's guard, #498),
        which stay in `_search` on the fit's `_mean_space_invalid` and
        `_link_domain_invalid`;
      - S04's lowering of the default start (`interior_start_intercept`), whose
        flag the overlay receives.
    - **A behaviour-neutral saving.** A fit without a mean space passes
      `first_halving=None` to the search, which reads that as depth 1, exactly
      what `mean_space_first_halving` returned for it.
27. **Fields that left the fit (29).** It has 92 fields, down from 118.
    - The QP (10): `has_constraints`, `A_all`, `b_all`, `abs_A_all`,
      `prev_active_set`, `committed_active_set`, `proposal_qp_converged`,
      `retained_qp_converged`, `_reported_qp_nonconvergence`,
      `_degeneracy_retry_failed`.
    - The policy (3): `_observed_newton_active`, `_stagnation_newton_available`,
      `_can_reuse_weighted_gram`.
    - The overlay (16): `_mean_space_newton_available`, `_mean_space_newton`,
      `_true_ratios`, `_true_stop`, `_true_confirmed`, `_true_stop_ratio`,
      `_last_true_residual`, `_held_rank`, `_judged_truncated`, `_last_truncated`,
      `newton_committed_rows`, `_boundary_rows`, `_published_truncated`,
      `_truncated_boundary`, `_row_set_curvature`, `_scop_columns`.
    - Added (3): `qp`, `curvature`, `mean_space`.
    - The fit's `observed_rows_invalid`, `true_mode_residual` and
      `_judge_mean_space` moved with their collaborators.
28. **The loop's order is unchanged:**
    1. `_converge`: the L16 criterion, then the step-rejected veto;
    2. `mean_space.judge`;
    3. `_terminate`: the QP feasibility and KKT veto, then the reason chain.

    After the loop, `_finish` runs F01, F02, F03 and then `mean_space.final`,
    which is F04 with the truncated relabel. `deviance_relative_change` stays in
    `run()`.
29. **The identity hazards hold.**
    - `final` compares the certificate's judged eta to `_finish`'s
      `eta_unclipped` by identity. That is still the retained state's own array,
      which `judge` hands the certificate.
    - `centred_rhs` keeps the system's arrays (`replace`), so the factor
      certificate's identity cache still hits.
    - `certify_centered_factor` and `retained_rank_info` are still bound by
      `partial`. A deferred callable reaches no fit, step or collaborator
      (`s3d_out/checks/cycle5.py`).

Where S4 (the stop rule and the shared line search) changed it:

30. **`_IRLSDirectConvergence`, in `_convergence` (the fit's `stop_rule`).** It is the
    stop rule: L16's criteria, the step-rejected veto and L18's veto and reason chain.
    - It is built in `_setup_loop`, from `max_iter` and the resolved `mode_bar`, where
      S21 bound the window. It keeps no reference to the fit.
    - **`judge(fit, it)` is the precedence, in one method:**
      1. `_criterion`: the criterion `convergence` names, then the step-rejected veto;
      2. `mean_space.judge(fit, stagnation_window)`, when the fit holds the overlay;
      3. `_terminate`: the feasibility contract's veto (`qp.feasible(beta)`,
         `qp.retained_converged`), then the reason chain.

      `run()` calls `stop_rule.judge(self, it)` once, between the search and the
      recorder. The judge's refused stops stagnate over the rule's window, which the
      rule now hands it instead of the judge reading a fit field.
    - **What it holds:** `stagnation_window`, `_mode_ratios` (its own), and
      `last_mode_residual` (the recorder's `profile_fit` reads it).
    - **The verdict stays in the fit's slots**, as the step's proposal does (point 2).
      The judge, the recorder and the loop read it there, and the judge still writes
      the rule's numbers after the rule when its certificate decides:

      | Field | Written by | Read by |
      |---|---|---|
      | `convergence_label`, `convergence_tolerance`, `convergence_value` | `_criterion`; then the judge under Newton steps | the recorder (`iteration`) |
      | `coef_change`, `dev_rel_change` | `_criterion` | the recorder (`debug_row`) |
      | `converged_this_iter` | `_criterion`; the judge; `_terminate` | the judge, `_terminate`, the recorder, `run()` |
      | `score_stagnated` | `_criterion`; the judge | `_terminate`, `run()` |
      | `termination_reason` | `_terminate`; after the loop, `_finish` (F03, F04) | the recorder, `_finish`, `_publish` |

    - **What `_criterion` reads:** `convergence`, `tol`, `max_iter`, `beta`,
      `intercept`, `retained`, `committed`, `step_rejected`, `dev`, `mu`, `eta`,
      `_score_centre`, `mode_bar`, `objective_merit_scale`, `curvature`, `recorder`,
      the step's `latent_change()` and `damped(fit)`, and `mode_residual`.
      **What `_terminate` reads:** `converged_this_iter`, `step_rejected`, `dev`,
      `max_iter`, `qp`, `beta` and `score_stagnated`.
    - **It is not pure (§6.6 point 4).** `_criterion` calls
      `fit.curvature.switch_to_newton(fit.recorder)` (the Gamma/log switch and its
      latch). The judge sets the overlay's `engaged`. The next iteration's `_working`
      and step read both. Neither is a field the rule caches.
    - **What stays the fit's:**
      - `mode_residual` (S07). It reads `coefficient_working_rows` and
        `penalized_mode_residual` from `_fit`, where tests patch them for the mode
        score together with L02's and the export rows' calls. Moved, those patches
        would silently stop reaching the mode score.
      - `mode_bar`, the bar of both certificates. `certify`, the judge and
        `profile_fit` read it with the fit's other settings.
      - `_score_centre`, a step output like `W` and `z`. The steps write it and
        `mode_residual` reads it.
31. **The criterion reads the iteration's base off the committed state.**
    - `beta_prev` and `intercept_prev` were `committed.beta` and `committed.intercept`,
      bound in `_working` from the same `committed`. `objective_prev` and `dev_prev`
      were the committed state's objective and deviance, advanced at L21 to the state
      just committed. The four fields and L21's advance are gone. The criterion and
      `deviance_relative_change` read `committed`, before L21 commits the retained
      state.
    - `score_stagnated` was read back from the last iteration's verdict. A stagnated
      verdict ends the loop, so every iteration found it `False`. The criterion now
      starts from `False`, and `_setup_loop` no longer sets it.
    - At S3d's head an assertion probe checked each of these at every read, over the
      harness and the focused test files: every read held by identity, 17,731 in the
      harness and 43,939 in the tests (`s4_out/probe/invariants.py`).
32. **`_IRLSDirectTrialSearch`, in `_search` (the fit's `trial_search`).** It is L10,
    L11 and L13–L15.
    - It is built in `_setup_loop`. It is `None` when the step has its own search: the
      SCOP step's latent search (L09) stays the step's (§6.6 point 6).
    - It holds L13's predicates `_mean_space_invalid` and `_link_domain_invalid`
      (#425, #498), formerly fit fields, and no reference to the fit.
    - `search(fit, it)` has the SCOP step's hook's signature. `run()` picks one of the
      two once, so the loop calls `search(self, it)` without a branch.
    - It reads `beta`, `intercept`, `committed`, `qp`, `newton_score`, `mean_space`,
      `y`, `weights`, `family`, `link`, `max_halving`, `objective_merit_scale`,
      `curvature`, `proposal_centred_intercept`, `evaluate_state`, `penalty_matvec`
      and `recorder`. It writes `retained`, `eta_unclipped`, `eta`, `mu`, `dev`,
      `beta`, `intercept`, `decision`, `n_halvings`, `step_rejected` and
      `proposal_state`, and the QP step's `retained_converged`.
    - It takes its mean-space hooks from the overlay's `line_search` and its
      constraint data from `qp`, as before.
33. **The feasibility contract's predicate is the QP step's `feasible(beta)`.**
    - `_QP_FEASIBILITY_TOL` moved to `_qp`, beside it. The package re-exports it from
      there for the editor.
    - The callers are where they were: the line search (L11: the committed state, the
      proposal and the trial predicate), the stop rule's veto (L18) and the final
      refusal (F03, `_finish`). The checks stay in the shared code. Only the test of
      `A beta >= b` is on the step's data.
34. **A rejected step no longer restores the QP step's active set.**
    - Every rejection ends the loop (`run()` breaks on `step_rejected`). On a rejected
      iteration nothing after the search reads the active set: the judge certifies
      only a stop, the veto reads `retained_converged`, and `_finish` reads the
      constraint system.
    - `committed_active_set`'s one reader was the restore, so its per-iteration copy
      (`list(active_set)` at the start of `propose`) went with it.
    - At S3d's head a probe set both fields, at the restore, to a value whose every use
      raises, and made `_working` refuse to start an iteration after a restore
      (`s4_out/probe/poison_restore.py`). The full harness ran 252 of 252 identical
      with 1 restore, and the focused test files 3,652 passed with 2. Two of the three
      restores would have changed the active set. Nothing read it.
35. **Fields that left the fit (9).** It has 85 fields, down from 92.
    - The stop rule's: `_mode_ratios`, `_stagnation_window`, `_last_mode_residual`.
    - The search's: `_mean_space_invalid`, `_link_domain_invalid`.
    - Copies of the committed state: `beta_prev`, `intercept_prev`,
      `objective_prev`, `dev_prev`.
    - Added (2): `stop_rule`, `trial_search`.
36. **The loop now reads:**
    1. `_working` (L02);
    2. `step.propose`;
    3. `search`, the step's or the shared one;
    4. `stop_rule.judge`: the criterion and the step-rejected veto, the mean space's
       judge, the feasibility contract's veto and the reason chain;
    5. `recorder.iteration`;
    6. `deviance_relative_change`, from `committed`;
    7. L21: break on a non-finite deviance or a rejected step; commit; break on
       convergence or stagnation.

    After the loop nothing changed: F01, F02, F03, then `mean_space.final` (F04 with the
    truncated relabel). F02 still reads `deviance_relative_change`.
37. **The identity hazards hold.**
    - `retained` is the trial cache's own state, and the judge hands `certify` its
      `eta_unclipped`, which `final` compares by identity.
    - No state is copied on the way into the stop rule.
    - A deferred rank-info callable still reaches no fit, step or collaborator, and no
      fit or unit outlives its return with the collector off; a unit that kept the fit
      would (`s4_out/checks/cycle6.py`).

Where S5 (finalisation per strategy, the leftovers and the cleanups) changed it:

38. **Two finalisers, and each step names its own.**
    - `_finalise` holds the centred finaliser, `_finalise_centred(step, fit)`: the fit's
      `_final_system` (now `_final_centred_system`, F08 and F09) and `_final_geometry`
      (F11), moved verbatim but for `self` -> `fit`, with `_feature_edf` and
      `_retained_rank_info` beside them (`s5_out/scripts/split_finalise.py` asserts each
      rewrite).
    - The QR, dense, QP and SCOP steps name it: `finalise = _finalise_centred`. A module
      function set as a class attribute is a method, so the step hands itself in and the
      finaliser reads its `data_rank`. The structured step's `finalise` is its own, as
      since S3b (point 8).
    - The interface gives `finalise` no default: a step that names none fails at its first
      finalisation. `run()` calls `step.finalise(self)` and hands what it returns to
      `_publish`, with no route branch.
39. **What a finaliser returns: `_IRLSDirectFinalGeometry`**, a frozen record (`slots`,
    `eq=False`) of what F10, F11 and F08 wrote and F12 to F14 read:

    | Field | Centred | Structured | Read by |
    |---|---|---|---|
    | `XtWX_beta` | raw `X'WX` | the block data operator | `_publish` (`return_xtwx`); SCOP's `postfit` (the raw Fisher Gram) |
    | `XtWX_S_inv_beta` | the slope pseudo-inverse | the profiled factor | `_publish` |
    | `log_det_H`, `reml_hessian_rank` | from the slope decomposition | from the profiled factor | `_publish` |
    | `reml_geometry_summary`, `retained_reml_decomposition`, `rank_info` | as before | `None` | `_publish` |
    | `p_eff` | `1 + sum(feature_edf)` | the factor's trace | `_publish` |
    | `structured_factor` | `None` | the profiled factor | the recorder's `disclose_backend` |
    | `centered_final`, `XtW1` | the final system, its raw `X'W1` | `None` | SCOP's `postfit` |

    - `_publish(it, final)`, `postfit(fit, result, final)` and
      `disclose_backend(fit, backend, structured_factor)` read it. SCOP's post-fit reads
      `sum_W` as `final.centered_final.sum_w`, the float the fit's `sum_W` held.
    - Why a record, when point 2 kept the proposal in slots: finalisation runs once per
      fit, so the record is one allocation, and it names the whole interface between the
      finalisers and publication. Each of the 13 slots it replaces had two writers, one per
      finaliser, and a reader that could not tell which had run.
    - **Fields that left the fit (13):** `centered_final`, `XtWX`, `XtW1`, `sum_W`,
      `XtWX_beta`, `XtWX_S_inv_beta`, `log_det_H`, `reml_hessian_rank`,
      `reml_geometry_summary`, `retained_reml_decomposition`, `p_eff`, `rank_info` and
      `structured_factor`. The fit has 72, down from 85.
40. **After the loop, in this order:**
    1. `_finish`: F01, F02, F03, then `mean_space.final`, which is F04 with the truncated
       relabel and compares the certificate's judged eta with `_finish`'s `eta_unclipped`
       by identity. `_finish` reads no route flag: `qp` and `mean_space` are the
       feasibility contract's and the overlay's own data, `None` on a fit without them.
    2. `_terminal_rows`: the export rows (F06), then the step's `terminal` (F05), which now
       reads them (point 41). F05 writes only the SCOP step's latent state, and F06 reads
       none of it, so the swap moves nothing.
    3. `recorder.profile_fit` (F07).
    4. `step.finalise(self)` (F08 to F11), whose record `_publish` reads (F12 to F14).
41. **Bitwise cleanups, one commit each:**
    - **The dense Newton step's `X~' u` once.** `centred_rhs` formed
      `centred_data_score(dm, newton_score, mean_x)` for the right-hand side and
      `newton_iterate` formed it again for the increment, from the same arrays. It is
      formed once, in `centred_rhs`, which returns it beside the system; the QP step drops
      it. On the binomial/log Newton route (`s5_out/checks/opcount6.py`, `msn_dense`, three
      Newton iterations) `centred_data_score` runs 14 times, not 17, and the fit executes
      259 fewer instructions; each saved call is a pass over the rows.
    - **`_working`'s reset of the iterate.** It copied `committed.beta` into `beta` (O(p)
      per iteration) and `committed.intercept` into `intercept`. Every step's `propose`
      writes both before anything reads them, and nothing between `_working` and
      `propose` reads either.
    - **Dead stores.** `_setup_loop`'s `XtWX_beta = None`: its one reader, `_publish`, ran
      only after a finaliser had written it. The old L05 block's
      `_final_penalized_operator` was already gone (S3b, point 10).
    - **The Fisher-data gate.** `get_centered_system` took the caller's Fisher data on
      `weighted_gram_reuse and not observed`. The policy computes the reuse flag with
      `and not observed`, after the latch, and `switch_to_newton`, the only other writer of
      either flag, sets `observed` and clears reuse together, so the second test never
      decided anything.
    - **`_even_exponent`'s zero branch.** Its one caller hands it the largest positive
      prior weight, and `_setup_design` refuses weights that are not finite, are negative
      or have no positive entry before binding the fit's weights, which nothing rebinds.
      The harness never took the branch (S4's branch coverage). It was redundant as well:
      `math.frexp` gives exponent 0 for zero, infinity and NaN.
    - **The per-iteration log line** passes lazy %-arguments, so INFO off formats
      nothing. The text is the same.
    - **The recorder's `np.isfinite(dev)`** runs only with the trace on; it guarded
      `emit_state_commit`, which returns at once without one.
    - **`t_eta` records what it names.** `irls_eta_s` was always 0. It is now the time
      forming the eta of every state the fit evaluates (from its coefficients, or a
      line-search trial's from its endpoints'), its clip and its mean:
      `_evaluate_irls_state` takes an optional `eta_clock`, which `evaluate_state` and the
      SCOP trials pass (`recorder.eta_formed`), and the callers time the predictors they
      form themselves (a centred state's, a shared-search trial's, a SCOP trial's on the
      ordinary block's centred rows). Wall-clock fields only.
    - **The SCOP terminal hook reads the export rows.** It formed Fisher rows at the
      retained model, and the fit formed the same rows from the same arguments as its
      export rows straight after, a duplicate pass inherited from master. A SCOP fit always
      forms export rows (it refuses `_return_working_system`), and `block_curvature` writes
      neither `W` nor `z`.
    - **The `_use_observed_newton` docstring** said SCOP keeps Fisher, false since #527.
42. **The stop rule's feasibility veto is gone.**
    - `_terminate` vetoed a stop on infeasible retained coefficients and again without the
      inner QP's KKT certificate. The certificate (`retained_converged`) is set only when
      the QP converged and the search kept its full proposal (point 32), and
      `solve_constrained_qp` reports `converged` only from a branch that has just tested
      the returned point with `_is_feasible` at its default `tol` 1e-12, which is
      `_QP_FEASIBILITY_TOL`, on the same `A` and `b`. So no infeasible retained state is
      certified, and the KKT veto holds it open (S4's probe saw the feasibility veto fire
      4 times, each with the KKT veto).
    - The feasibility test stays: it names `constraint_infeasible`.
    - `test_an_infeasible_retained_qp_state_is_never_kkt_certified`
      (`test_constraint_fit_feasibility.py`) pins the invariant: the two tolerances are
      equal, and at every verdict of healthy QP fits, a warm start outside the set and tiny
      accepted steps that keep the retained state outside, no infeasible retained state is
      certified. Mutations: the certificate handed to any accepted step fails it at the
      invariant, and a contract tolerance of 1e-13 at the tolerance check.
43. **Leftovers deleted.**
    - `_setup_loop`'s `deviance_relative_change = inf` seed: its one reader is `_finish`,
      and the loop sets it in every iteration before any break; the loop runs at least
      once. The rule its comment gave, why the change is taken before the commit, moved to
      the line that takes it.
    - `fit_irls_direct`'s docstring pointed at `_fit_irls_direct_once` for the parameters;
      it points at `_IRLSDirectFit`.
    - Kept, with their readers: every name `__init__` re-exports has an importer in `src`
      or `tests` (17 of 17); `_fit_irls_direct_once` is `test_no_auto_fallback`'s spy
      point; `_build_penalty_matrix` is imported by four test files and `_safe_decompose_H`
      by `reml.efs` and two; the working-system guards (`_last_working_centered is None`,
      the structured `_last_system is None`) are invariants a probe forces (point 44).
44. **The finalisation's branches the harness does not reach** (its branch coverage at
    S5's head) each have a probe, run under branch coverage on S4's head and on S5's
    (`s5_out/probe/branches_s5.py`, `targets_s5.py`): 29 records, identical on both
    heads, and 28 of the 29 missed lines and arcs executed on both:
    - the EDF's scaled trace (subnormal-weight binomial/log fits; direct calls for a
      Hessian with no finite scale and for a scaled Gram that refuses, which takes the
      factor);
    - the data rank's two factor routes, unpenalised (the fit's own certificate) and
      penalised with the Gram and the grey-subspace certificate refusing;
    - the working-system guards of both finalisers, forced, and the fREML iteration that
      keeps its system;
    - publication's SCOP shapes with and without the data Gram;
    - no caller profile; a candidate without REML geometry, on both routes.

    The one arc not run is in unmoved recorder code: `disclose_backend`'s test of a
    structured factor without a border certificate, which none of the three profiled
    factors is (each assigns a `BorderCertificate`).

    Of 23 finaliser mutants (`s5_out/scripts/make_mutants_s5.py`), the harness or the
    focused tests catch 18: every wrong or missing finaliser, both working-system flags,
    the record's fields, the dropped relabel, the copied judged eta, the boundary rows, the
    F01 deviance, the F03 KKT refusal, the QR rank deferred, the dropped `cache_out`
    entries, and the SCOP post-fit's Gram (the harness; the SCOP tests miss it) and
    publication's SCOP tuple (the SCOP tests; the harness never returns it). Five are
    missed, and none is equivalent in principle:
    - four F-block reorders (F04 before F03, F03 before F02, F02 before F01, F04 first).
      Two F-blocks interact only when both act on one fit, and no fixture has such a pair:
      a Gaussian identity fit (F01) with a separation signature (F02); a constrained fit
      with an extreme weight ratio that runs out its budget, where F02 reads the
      `converged` F03 clears; a constrained binomial/log fit refused by the QP contract
      and at the mean-space boundary (F03 and F04);
    - the EDF's cut at `100 eps`: no fixture has a coefficient EDF in `(0, 100 eps)`.
45. **The identity hazards hold.**
    - `final` still compares the certificate's judged eta with the retained state's own
      `eta_unclipped`, which `_finish` hands it; no state is copied after the loop.
    - With the collector off, no fit, step, collaborator or record outlives its return on
      any route of `cycle6.py` (now counting `_IRLSDirectFinalGeometry` too), and each of
      the six units that keeps the fit is seen. A deferred rank-info callable reaches no
      fit, step, collaborator or record.
    - Against S4 (`opcount6.py`), every route executes fewer instructions (gram 11,273 to
      11,229; SCOP Poisson 20,875 to 20,600; `msn_qp` 49,297 to 48,686) and a few more
      Python calls, `eta_formed` once per state evaluation.

Tests patch `superglm.solvers.irls_direct._fit`, the module whose globals the fit reads.
A name that a strategy module reads is patched in that module: since S3b the structured
factor build, `_build_iterate_factor`, `keep_narrow_cap` and the score centre's
`centre_offset_mean` are `_structured`'s, and tests read the two errors there.  Since S3c
the SCOP step's are `_scop`'s: `_select_irls_trial` (SCOP's search; the shared search
still reads `_fit`'s), `_evaluate_scop_trial`, `_latent_step_depth`,
`_centred_system_matvec` and `_raw_centering_well_scaled`, and the SCOP tests patch and
read them there.  Since S3d:
- the QP step's `solve_constrained_qp` (and `_solve_constrained_qp_with_cold_retry`) are
  `_qp`'s;
- the dense step's `decompose_gram_if_authoritative` is `_dense`'s, and the three tests
  that refuse the loop's Gram patch it there as well as in `_finalise` (since S5; `_fit`
  before), whose finaliser reads it;
- the certificate's `truncated_direction_ratio`, `row_set_residual`, `row_sets` and
  `null_basis_angle` are `_mean_space`'s, and the package re-exports `_SUBNORMAL_SPACING`
  and `_underflow_allowance` from there.

Since S4 the shared search's `_select_irls_trial` is `_search`'s (eight sites), and the stop
rule's `rate_exhausts_budget` and `_irls_objective_relative_change` are `_convergence`'s. The
tests patch and read them there; `_fit` no longer imports them, so a stale patch at `_fit`
raises.

Since S5 the centred finaliser's `decompose_gram_if_authoritative`, `decide_data_rank`,
`grouped_weighted_factor` and `grouped_projected_factor` are `_finalise`'s, and the seven
test files that patch them (`test_tsqr_certificate`, `test_irls_direct`,
`test_factor_smooth_structured_parity`, `test_gamma_fisher_data_reuse`,
`test_robust_solve`, `test_data_rank_grey_subspace`, `test_discarded_fit_rank_info`)
patch them there; `_fit` no longer imports them.

Since the port of the finalise reuse (after S6), the dense step's iteration and the centred
finaliser's REML slope take the Gram decomposition through the fit's
`authoritative_gram_rank`, a `partial` of `_fit._authoritative_gram_rank` over a one-entry
memo, bound in `_setup_systems` beside `certify_centered_factor` and, like it, never a bound
method.  Their `decompose_gram_if_authoritative` is therefore `_fit`'s: `_dense` no longer
imports it, and `test_robust_solve`, `test_gamma_fisher_data_reuse` and
`test_factor_smooth_structured_parity` patch it at `_fit` where they patched `_dense`.  The
finaliser's own calls (the coefficient rank and the EDF's rescaled Hessian) stay
`_finalise`'s.

What stays at `_fit`:
- `build_centered_system` (17 sites and `test_lookup_table_gram`'s string path), because
  the centred system did not move;
- `coefficient_working_rows` and `_has_constant_irls_weights`, because the policy lives in
  `_fit`;
- `penalized_mode_residual`, whose two patch sites target `mode_residual`, which stays a
  method of the fit (point 30).

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
