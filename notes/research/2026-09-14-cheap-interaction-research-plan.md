# Cheap interaction research and implementation plan

> **For agentic workers:** Use `superpowers:executing-plans` for a bounded
> milestone, or `superpowers:subagent-driven-development` when independent
> agent work is authorized. Track execution with the checkboxes below. This
> document plans research; unchecked gates are not completed capabilities.

**Goal:** Find regimes in which useful, inspectable interactions require much
less complete fitting time and peak RSS than ordinary full tensors, with
explicit approximation and numerical error contracts.

**Architecture:** First measure compressibility of existing useful fitted
surfaces. Then fit compact representations directly, starting with nested
selected products in a fixed Gaussian problem. Promote local refinement,
learned factors, or iterative solvers only when the preceding measurements
identify a reason to build them.

**Tech stack:** Existing SuperGLM, NumPy/SciPy, Python 3.12+, the development
environment on Python 3.13, Matplotlib, and the pinned Lean/Mathlib project in
`docs/research/lean-gaussian-certificate`.

**Spec:** The approved
[adaptive-interaction design](2026-09-13-adaptive-interactions-research-design.md),
the [representation analysis](2026-09-13-cheap-interaction-representations.md),
and the user's 2026-09-14 request to test the cheap-representation hypothesis.
This plan refines their execution order using the subsequent real-data trials.
Audit head: `7a317569`. No new fit, proof, or speedup accompanies this plan.

## Global constraints

- Work in `.worktrees/adaptive-interactions`; preserve unrelated user changes.
- Keep mathematical findings, failed conjectures, proof sources and receipts
  under `docs/research`. Keep prototypes and timing outside production paths.
- Use `gpt-6-astra` at `max` for mathematical agents and no more than three
  concurrent subagents. Give each a bounded deliverable and report its status.
- Retain every observation within each declared fit. Training subsets belong
  only to explicitly labelled scaling experiments.
- Never relax solver tolerances to obtain a successful benchmark. Numerical
  tolerances derive from dimensions, dtype epsilon, norms and conditioning or
  certified error bounds.
- Compare complete fits, numerical outputs, actual backend dispatch, peak RSS
  and retained payload. Charge every search attempt and certification pass.
- Run timed fits serially in fresh single-thread workers with owned deadlines.
  Apply the repository [timing policy](../development/cost-and-timing.md).
- Preserve source/data/model hashes and old raw artifacts. Previously examined
  test sets are development evidence, including when repartitioned.
- No public API or SuperLSS naming decision is part of this research plan.
  Release, dependency and version changes are outside this task.

## 1. The hypothesis and what a proof could establish

The hypothesis is that useful interaction functions often occupy a compact
part of a much richer tensor space. That compact part might consist of a few
products, a few learned separable components, a small locally refined mesh,
or factors shared across several pairs. These are different assumptions.

The research question is: **how does the smallest adequate representation grow
with accuracy requirements, sample size, selected pair count and signal
complexity, and can we find and fit it cheaply?** There is no presumed universal
ten-times-additive bound. We will report curves around 2, 5 and 10 additive fit
times and retain absolute seconds.

| Claim | What can establish it | What would remain unknown |
| --- | --- | --- |
| A compact map represents a specified function and penalty correctly | Exact identities, Lean proofs, implementation reconstruction tests | Whether data need a compact function |
| A candidate is close to a fixed rich Gaussian optimum | Full residual bound with verified numerical premises | Population prediction, a different smoothing optimum, interval coverage |
| A function class admits a given approximation rate | A theorem under stated regularity, rank or coefficient-decay assumptions | Whether those assumptions describe the real workloads |
| An implemented action has a particular operation/storage count | Algorithm derivation, loop/allocation audit and structural regression tests | Complete fit time if iteration or smoothing costs grow |
| A procedure is useful and faster on real problems | Frozen evaluation protocols and complete resource measurements | Universality across all data distributions |

One useful conditional example is a marginal-product metric. For a centered
surface `f(x,z)=b_x(x)^T C b_z(z)` and positive-definite mass matrices `M_x,M_z`,
put `D=M_x^(1/2) C M_z^(1/2)`. Exact singular-value truncation gives

\[
\|f-f_r\|_{L^2(\mu_x\otimes\mu_z)}^2
  =\sum_{i>r}\sigma_i(D)^2.
\]

This statement uses a specified product measure. It is not automatically an
error bound on correlated observed pairs or on the joint penalized fit.
Singular mass matrices require a justified support quotient; no arbitrary
ridge may silently define a different metric. Rapid singular-value decay is
something to measure, not a consequence of calling a function smooth.

Shared factors offer a stronger conditional scaling result. For fixed small
rank `r`, marginal functions `h_jt`, and signs `sigma_t`,

\[
\sum_{t=1}^r\sigma_t\sum_{j<k}h_{jt}(x_j)h_{kt}(x_k)
=\frac12\sum_{t=1}^r\sigma_t
\left[(\sum_jh_{jt}(x_j))^2-\sum_jh_{jt}(x_j)^2\right].
\]

With `K=sum_j k_j` marginal coefficients, dense value/gradient passes can cost
`O(nKr)` and factor storage `O(Kr)`. These are conditional action counts from
the existing analysis. Shared basis storage, optimization, penalty evaluation,
rank selection and requested outputs remain additional costs. An arbitrary
selected-edge mask generally restores an `O(nMr)` edge accumulation after
marginal evaluation. Individually low-rank edges need not share low global
rank. We are not proposing explicit fitting of the entire pair universe.

Rügamer's factorized spline construction provides relevant prior art, including
factor-count-dependent scaling. It uses its own penalization and optimization
scheme; the paper reports slow convergence for investigated block-coordinate
variants. It does not prove SuperGLM's complete REML cost.
[Primary paper, Sections 4.1 and 4.3](https://proceedings.mlr.press/v238/ruegamer24a/ruegamer24a.pdf)

Sparse-grid rates have regularity assumptions too. The classical result for
piecewise-linear sparse grids assumes bounded second mixed derivatives.
The current tensor penalty on pure directional derivatives does not by itself
establish that assumption or the same rate.
[Bungartz and Griebel](https://www.cambridge.org/core/journals/acta-numerica/article/abs/sparse-grids/47EA2993DB84C9D231BB96ECB26F615C)

## 2. Roadmap mapping and priority

| Roadmap item | Contribution to this plan | Present evidence | Promotion gate |
| --- | --- | --- | --- |
| **C21: economical spline interactions** | Main representation work: selected products, independent factors, then shared factors | Algebra/cost analysis and useful fitted surfaces; no compact direct fitter | Direct compact fits preserve declared accuracy and reduce whole-fit cost or memory |
| **C16: adaptive hierarchical splines** | Allocate detail locally when measurements favor that representation | Centering, nesting and penalty design; no adaptive implementation | Verified transfer/pullback/adjoints, then an advantage over selected products and uniform tensors |
| **C15: matrix-free EFS/REML** | Avoid dense coefficient matrices if compact models still make them dominant | Exact operator identities and cost audit; no production iterative backend | Full coefficient and smoothing costs, numerical certification and memory beat the relevant direct baseline |
| **C18: multigrid/Krylov recycling** | Reduce iterative work using a valid hierarchy or reusable subspace | Analysis; depends on an iterative backend for the proposed recycling work | Measured iteration/preconditioner benefit including setup and invalidation costs |
| **C9: bounded scalar discovery** | Choose pairs and capacity together after compact fitting is usable | Training-only proposals plus bounded validation/refits; known blind spots | Gains survive the full search budget and fresh evaluation; no exhaustive-recovery claim |
| **C1: compressed row operations** | Preserve/reuse marginal preparation and compatible row-side compression | Existing discrete machinery and one exact tensor support-handoff improvement | New representation composes with it without changing the target or rebuilding expensive blocks |
| **C4: shared smoothing/effects** | Optional smoothing pooling for factor models | Separate candidate in roadmap | Explicit model/penalty contract and accuracy comparison; shared lambda is not shared coefficients |
| **C24: compressed coefficient matrices** | Only a possible later alternative if the coupled Hessian has suitable block ranks | Speculative | Measure Hessian block ranks; low-rank surfaces do not imply a low-rank Hessian |

The immediate representation priority is **C21 diagnostics, then a small
C21/C16-compatible linear subspace fitter**. Full C16 mesh machinery and
C15/C18 are conditional branches, not prerequisites to learning whether
compression helps. General distributional C9 and coupled LSS fitting remain
later scopes. Row subsampling/exactification would require a distinct C20
contract and is not the first method.

The existing exact Gaussian tensor cross-product reuse candidate remains a
separate performance track. It can improve the ordinary tensor baseline and
must be included in later comparisons if implemented. It does not answer the
representation question and need not block the first diagnostic.

## 3. Existing evidence and the first workloads

The [broad trial](2026-09-14-broad-interaction-trials.md) supplied useful selected
groups on ten of twelve real sources, with no more than four pairs per model.
The gain belongs to each selected group, not to each individual plotted term.
The models use a small k=4/6 menu; that menu cannot establish the savings
available for rich interactions.

| Workload | Role in the next experiment | Limitation to preserve |
| --- | --- | --- |
| Airfoil, selected two-pair model | First saved-model compression diagnostic; sign-changing structure | Grouped split; already examined test; small marginal widths |
| King County, selected four-pair model | Joint compression with correlated features and repeated parents | Temporal/property exclusions; group gain does not isolate individual pairs |
| Concrete, selected four-pair model | Different smoothness/local-detail pattern and numerical stress | Two separate one-pair fits were refused; those outcomes remain in the record |
| California housing latitude/longitude, `rows20` and `rows30` | Wide tensor comparison with 361/841 interaction coefficients | Existing bases are not a nested k ladder; historical test has informed design |
| Bike hour/working-day and temperature/hour | Later mixed-type and non-Gaussian transfer check | Saved hour is categorical: categorical-by-categorical and spline-by-categorical Poisson terms need separate contracts |
| Synthetic additive-only and known interaction laws | Detect invented signal, coarse-screen misses and representation limits | Mechanism controls, not additional real datasets |

Start with the ten terms already plotted for Airfoil, King County and
Concrete, preserving each complete selected model. Reuse saved model/input
identities. A housing prediction archive is not a fitted coefficient snapshot:
if no verified snapshot is available, acquire one bounded reference fit for
each requested housing case and record its full cost. No reference refit is
hidden inside a purported cheap-fit measurement.

## 4. Milestones and decision gates

### M0. Freeze the experiment and extract a faithful reference

- [ ] Save model/data/source identities, feature units, centering, knots,
  coordinate maps, penalty components, smoothing values and split roles.
- [ ] Reconstruct each contribution in the actual centered marginal basis.
  After reconstructing any split-group coefficients, current
  `TensorInteraction.score` applies `beta_eff = _R_inv @ beta_full` when the
  map is present, then reshapes to `(_p1, _p2)`. Do not invert `_R_inv`.
- [ ] Replay the complete saved predictor and selected term contributions.
  The scientific quantity is the link contribution, not exponentiated
  "relativity" for these Gaussian identity-link cases.
- [ ] Distinguish original-input prediction geometry from the discretized
  fitting design. A certificate must name exactly which compiled target it
  bounds. Keep discretization error separate from representation error.

**Deliverable:** immutable extraction receipt and numerical replay evidence.
**Stop/refuse:** missing identity, unsupported coordinate map, changed target,
or unexplained replay discrepancy. No SVD interpretation before replay works.

The [pilot implementation plan](2026-09-14-interaction-compressibility-pilot-plan.md)
specifies the first bounded software task in detail.

### M1. Measure compressibility before inventing a new fitter

- [ ] For each centered coefficient matrix, evaluate every feasible truncation
  rank, including zero and full rank. Record actual factor entries, not an
  assumed saving: `r(k_x+k_z)` can exceed `k_x k_z` for small matrices.
- [ ] Compute singular spectra in declared training marginal-product geometry.
  Also measure errors on actual paired training and validation observations.
  Do not interpret singular values of raw, arbitrarily scaled coefficients.
- [ ] Compare rank compression with nested selections of marginal penalty-mode
  products. Keep the rich basis and its penalty normalization frozen.
  A coefficient-based best-subset diagnostic may use the rich fit, but must
  be labelled an oracle diagnostic and charged as such.
- [ ] Compress terms one at a time and then together. Preserve mains for the
  diagnostic, and record both component error and total predictor/loss change.
  A later direct fit must let mains readjust jointly.
  Keep each edge's product-measure error separate; their sum is not an
  aggregate predictor error without the required orthogonality assumptions.
- [ ] Use additive-only, separable, several independent separable components,
  localized bumps and diagonal-ridge controls. A localized separable bump can
  itself be rank one; locality is not a synonym for high rank.

**Deliverable:** curves of representation size against function discrepancy,
development loss and retained payload, with all attempted ranks reported.
No fit-time speedup is claimed from post-fit compression.

**Decision:** prioritize selected products if a small linear subspace works;
prioritize independent learned factors if rank compression is substantially
better; prototype a local hierarchy if error is concentrated spatially and
the simpler representations remain expensive. Failure on one surface is
evidence about that surface and metric, not an impossibility result.

As provisional engineering guides, display 90%, 95% and 99% retention of a
positive reference improvement over additive. Use 95% as the first decision
point, subject to uncertainty, and report the whole curve. These are
statistical approximation allowances, not numerical tolerances. For
`G=L_add-L_rich>0`, gain retention is `(L_add-L_compact)/G`; if `G` is absent,
small or uncertain, use absolute loss differences and keep additive eligible.
Set the final application margin before a fresh confirmatory evaluation.

### M2. Directly fit a nested selected-product model at fixed smoothing

This is the recommended first fitting prototype. It stays linear in the
coefficients and reuses the existing fixed-Gaussian analysis.

- [ ] Compile a small Gaussian rich problem with known `X,W,S,b` and a verified
  identifiable coordinate system. Begin with an uncompressed dense oracle
  fixture; establish C1/discrete target equivalence separately before claiming
  a result for the saved production discretization.
- [ ] Define an injection `P_A` from active coefficients into rich coordinates.
  Include all main/nuisance directions and every penalty-null direction.
  Use `X_A=X P_A` and `S_A,j=P_A^T S_j P_A` for every component.
- [ ] Fit all active coefficients jointly. Start with null directions and a
  deterministic low-frequency product set; use training residuals to propose
  additional products. A fixed manifest resolves score ties.
- [ ] Check the full rich residual, including active-solve error and all
  omitted directions. Local frontier scores guide expansion; they cannot
  certify an unexamined complement.
- [ ] Add products in bounded batches, refit, and stop when the certified
  approximation budget passes or the declared budget is exhausted. Preserve
  the latter as `budget_exhausted` or `uncertified`, not success.
- [ ] In the direct candidate worker, never construct a full tensor design or
  Gram merely to fit the smaller model. Stream required rich residual actions
  and count their cost. A small dense oracle belongs in a separate worker.

**Deliverable:** an executable fixed-target reduced fit, proof/implementation
contract ledger, and a size-error-cost comparison with the full reference.
The initial fixed lambda is an external benchmark input. If a usable
procedure learns it by first fitting the rich model, charge that fit.

**Gate:** the compact candidate meets its stated numerical/approximation
contract and improves the measured resource frontier. If full residual checks
consume the saving, record that result and compare with a separately labelled
predictive compact-model procedure. Do not weaken the certificate to pass.

### M3. Build only the representation branch supported by M1/M2

| Branch | Bounded first build | Additional requirements | Pivot evidence |
| --- | --- | --- | --- |
| Local C16 hierarchy | One two-dimensional dyadic hierarchy and fixed fine reference | Product centering, nested transfer, nullspace/quotient, component penalty pullbacks, adjoints and quadrature | Closure/marginal corrections or global detail eliminate the savings |
| Independent factors | One selected pair with `C=UV^T`, then a small selected group | Original tensor penalty evaluated on `UV^T`; factor nonidentifiability, stationarity and restart costs | Ranks or optimization passes grow enough to erase savings |
| Shared factors | Controlled compatible cross-edge signals, then a bounded real group | Shared-rank compatibility, signed/PSD choice, mask cost, penalty pooling and stable accumulation | Low individual ranks do not yield low shared rank |

For the local branch, use the
[existing THB design](2026-09-13-adaptive-interaction-representation.md).
Its physical mass/stiffness penalty is not automatically equivalent to the
current cardinal natural-spline penalty. Compare matched uniform coarse/fine
models in the new space before a predictive comparison with current tensors.
Degree-six cubic mass products need exact integration or four Gauss points per
span. Admissibility/closure counts and centering corrections belong in memory
and action costs.

For factor branches, keep the existing tensor function penalty for the first
matched experiment. A separate sum of factor roughness penalties changes the
target. Nonconvex factor stationarity is not global optimality. Rich residual
certification may still assess a lifted candidate for a fixed Gaussian target,
but independent factor optimization has no automatic certificate of success.

### M4. Restore automatic smoothing and measure complete cost

- [ ] Compare fixed-parameter and automatically smoothed fits separately.
  Charge basis/rank choices, all lambda evaluations, inner solves, line
  searches, finalization and any required reference fitting.
- [ ] Treat reduced-space REML as a different model unless omitted-direction
  determinant/trace corrections and approximation errors are justified.
  A fixed-lambda mean bound does not certify its optimum or EDF/covariance.
- [ ] Fit the best compact candidate directly without importing reference
  coefficients or factors. Use bounded training initialization/rank menus.
  Keep a full-tensor control and simple smaller-tensor controls.
- [ ] Add the Bike Poisson/mixed-term check only after the Gaussian result;
  bound inexact inner solves and verify the full fitting acceptance rules.
- [ ] Report prediction-only and requested-inference workloads separately.
  Data-selected representations do not inherit ordinary post-selection
  interval coverage. If inference uses a subsequent rich refit, charge it.

**Deliverable:** matched whole-fit and end-to-end cost/accuracy frontiers on
several mechanisms and real cases, with an explicit inference scope.

**C15/C18 decision:** profile the winning compact models. If coefficient
factors remain dominant, use the existing coupled operator identity to build
a bounded iterative Gaussian backend. Account for preconditioner setup,
iterations, reuse invalidation, smoothing traces/log determinants, certification
and requested covariance actions. A cheap `Hv` is insufficient. Keep the
direct solver as a measured alternative.

### M5. Test scaling and reconnect discovery

- [ ] Run the independent scaling ladders in Section 6; do not replace them
  with one large model or infer an exponent from two timings.
- [ ] Reintroduce bounded training-only pair proposals and validation-selected
  capacity. Charge proposal generation, rejected candidates, exploration of
  finer directions, all joint refits and stopping/model-size decisions.
- [ ] Retain zero interactions as an outcome and include pure-interaction
  alternatives with weak main effects. Existing GBM proposals can miss
  balanced XOR; PSST EDF rungs are not basis-resolution choices.
- [ ] Freeze the method and evaluation protocol before obtaining confirmatory
  outcomes. Compare useful prediction at equal total budgets as well as cost
  at matched accuracy.

**Deliverable:** a bounded scalar C9 procedure whose cost includes discovering
and fitting its compact interactions, plus evidence describing when it works.
This does not complete general distributional structure discovery.

## 5. Lean and numerical contracts

Lean is valuable at the mathematical interfaces where a false identity could
invalidate many experiments. It is not required before looking at a spectrum
or timing an explicitly uncertified prototype. A proof does not show that
Python executes the stated algorithm or that computed inputs meet assumptions.

### What is already checked

[Certificate.lean](lean-gaussian-certificate/Certificate.lean) contains
`SuperGLM.residual_identity`, `quadratic_gap`,
`residual_energy_of_left_inverse` and `quadratic_gap_residual`.
They assume exact stationarity, symmetry where needed, and an exact supplied
left inverse for the inverse identities. They do not construct or certify a
computed inverse.

[InteractionOperator.lean](lean-gaussian-certificate/InteractionOperator.lean)
contains `coupled_hessian_action`, `group_row_accumulation`,
`group_penalized_row_accumulation` and
`dropping_cross_terms_changes_action`. They preserve cross-group terms in
finite-real algebra, including unequal group widths. The final theorem is
an explicit counterexample to dropping those terms.

The [tutorial](lean-gaussian-certificate/README.md) also has two checked
elementary examples. Existing receipts record successful checks of these
sources. None proves SPD/coercivity, the Schur upper bound, weighted prediction
error, Galerkin optimality, rank approximation, floating-point execution,
complete REML convergence or runtime.

### Contracts to add at each gate

| Contract | Exact mathematical obligation | Implementation obligation | Lean priority |
| --- | --- | --- | --- |
| P1: reduced-space map | Function preservation and quadratic/component penalty pullback through `P_A` | Replay coordinates and all terms; refuse unsupported maps | First new proof, before calling M2 a certified reduction |
| P2: nested minimization | Nested range inclusion; Galerkin residual and nonincreasing exact quadratic optimum | Retain active residual and joint main-effect adjustment | With P1; no claim of monotone validation loss |
| P3: residual certificate | SPD plus a valid comparison/Schur bound gives an upper bound on rich energy and weighted training prediction error | Verified residual enclosure, positive lower bounds, null correction and solve errors | Required before a Lean-backed numerical-certificate claim |
| P4: local hierarchy | With one frozen fine-space centering projection `Q`, `E_A=E_B T` implies `Q E_A=Q E_B T`; identified quotient and penalty equivalence | Rank evidence, adjoints, stable centered actions, integration/closure checks | Before the C16 branch claims certified transfer |
| P5: factor algebra | Exact factor predictor/penalty identities; shared-factor identity under its mask/model assumptions | Gradient/stationarity checks, factor/accumulation rounding and charged optimization | Before promoting a factor kernel; SVD-rate formalization can wait |
| P6: iterative backend | Full coupled operator, residual-to-error implication under verified geometry | Preconditioner properties, stopping enclosures and reuse invalidation | Extend existing operator proof when C15 is built |

For P3, the current
[penalty-null Schur bound](2026-09-13-adaptive-gaussian-error-bounds.md)
is a handwritten derivation, not a Lean theorem. One bounded formalization can
start with a conditional comparison. Fix the symmetric target Hessian and
stationary reference `H beta_star=b`. Define `e=beta_star-beta_tilde`,
`r=b-H beta_tilde` and `E=sqrt(e^T H e)`. In those same rich coordinates,
verified `H >= alpha G` in Loewner order, `G` positive definite and `alpha>0` imply

\[
E\leq(d_{\rm up}+\zeta)/\sqrt{\alpha},
\qquad \|W^{1/2}X(\beta^\star-\widetilde\beta)\|_2\leq E.
\]

Here `d_up` bounds the computed residual's `G^{-1}` norm and `zeta` bounds
residual-evaluation error in the same norm. The implementation must supply
those bounds and a justified matrix comparison. If it obtains `G` from the
Schur construction, that construction and its numerical admission also need
evidence. Merely assuming the desired inequality in Lean is not certification
of a run. All active/null residuals and cross corrections survive rounding.
The prediction implication also uses `H=X^T W X+S`, nonnegative observation
weights and a positive-semidefinite penalty. These are explicit premises.

Keep three quantities separate: a chosen approximation budget for the fixed
target, the arithmetic error enclosure required to justify that budget, and a
statistical loss allowance assessed on held-out data. Epsilon-derived bounds
control computation; they do not choose the scientific accuracy requirement.

For each promoted contract save its theorem statement, source identity,
assumptions, `#print axioms` output, pinned Lean/Mathlib versions, build log,
implementation mapping and adversarial test. Do not admit `sorry`, `admit`,
`sorryAx` or new unproved project axioms as finished proofs. Deliberately
invalid tutorial files stay outside the build and must fail when checked.
Use meaningful mutation tests, such as dropping a cross block, transposing a
residual map incorrectly, or omitting a null correction.

Full approximation-rate theory is a follow-up only after the observed
compression mechanism is known. A conditional theorem, a counterexample or
an accurately documented failure are all research outputs.

## 6. Scaling and resource experiment design

Record `n`, raw predictor count `d`, candidate count, selected pairs `M`,
marginal widths `k_j`, retained product count `R`, per-edge/shared ranks,
total coefficients `P`, nullity, smoothing dimension `q`, and iteration/pass
counts. These dimensions cannot be substituted for each other.

| Axis | Initial ladder | Hold fixed / purpose |
| --- | --- | --- |
| Rows `n` | 2k, 10k, 50k, 100k, then available full training size | Fixed features, selected pairs, representation and validation; keep groups intact |
| Potential predictors `d` | 8, 16, 32, 64, 128, then available width | Fixed rows and bounded selected pairs; separate nuisance-feature discovery cost |
| Selected pairs `M` | 0, 1, 2, 4, 8, 16, 32 where feasible | Fixed widths/rows and declared signal law; spread versus hub graphs |
| Rich marginal width `k_j` | Effective widths 5, 9, 17, 33 where admitted, built by verified nested refinement | Freeze each level's main/basis/penalty geometry within rich-versus-compact comparisons; record actual widths when boundary constraints change them |
| Rank / retained products | All small feasible ranks; geometric product budgets up to full | Plot error and cost together; report required rank as accuracy tightens |
| Smoothing dimension `q` | Fixed lambda, independent automatic lambda, explicitly pooled alternative | Separate tuning cost from representation size; pooled smoothing is a different model |

These are staged ladders, not a Cartesian sweep. Pilot one modest case per
axis, inspect allocations/refusals, then extend only an informative branch.
The width ladder changes the rich reference; the rank/product ladder changes
the approximation within one frozen reference. Independently placed quantile
knots are separate model comparisons and do not count as nested refinement.
For synthetic `M` sweeps, include a fixed-total-signal-variance series so adding
pairs does not manufacture an easier gain comparison. Also report an explicit
fixed-per-edge-signal series if the effect of growing total signal is wanted.
Cross-edge compatibility and joint conditioning are controlled separately.

Initially cap each fit worker at 180 seconds and each staged batch at 1,800
worker seconds. Persist all attempted outcomes, including timeouts and parent
cleanup. Larger admitted workloads need a recorded budget revision before
execution. Use three serial repetitions only for shortlisted timing comparisons,
with paired/reversed order and host activity records. On a contended host,
retain counts/allocation evidence and mark absolute speed claims unmeasured.

Report at least:

- Whole public-fit time and total procedure time, each divided by its matching
  additive baseline, plus absolute seconds and failed-search costs.
- Process peak RSS through fitting and through all requested work, retained
  array/buffer payload, largest temporary and ownership/lifetime accounting.
- Actual matrix/group classes, backend, phase/call counts, solver/outer
  iterations, restarts, full residual passes and preconditioner/trace work.
- Reference/compact loss, component and aggregate prediction discrepancy,
  certificate status, numerical refusal and termination reason.

Baseline fits use the same parent functions, observations, target/link,
weights and output requirements. A smaller marginal basis that also changes
the main effects needs a separate additive control. Small-k ordinary tensors,
uncompressed rich tensors and the compact candidate are separate arms.

Large real sources already available in the corpus can supply later axes:
YearPrediction for rows, CT/Blog for wider predictors, and ULB fraud for a
large imbalanced classification workload. Their adapters, feature exclusions
and valid split contracts must be verified before use; catalogued data are
not automatically fit-ready. SGEMM's all-numeric interaction model is a useful
resource control but does not demonstrate rich spline compression.

## 7. Validation and paper evidence

All previously inspected Airfoil, Concrete, housing, Bike and broad-trial test
outcomes are now development evidence. Freeze the experimental family before
using genuinely unexamined sources or future time blocks for confirmation.
New random splits of already examined sources assess robustness but do not
erase earlier design feedback. Respect household/property, subject, formula,
mixture and temporal grouping, including the existing exclusion rules.

For each source use train-only preprocessing/proposals and inner validation
for pairs, basis size, rank and smoothing policy. Persist choices before the
outer evaluation. Report paired loss differences and uncertainty using the
appropriate sampling unit. A bootstrap of fixed-model predictions quantifies
only that evaluation uncertainty; it does not include model-selection or
training instability. Those need repeated complete procedures or an explicit
conditional scope. Any multi-source aggregate states its weighting rule and
retains per-source failures and regressions.

Create a claim ledger linking each proposed paper statement to one of:
published theorem, new handwritten derivation, Lean theorem, numerical
admission contract, operation-count argument, or empirical receipt. Record
assumptions, failure regimes and exact revisions next to the claim. Preserve
spectra, size-error-cost curves, total search costs and negative controls.
Do not label an already published factorization identity a novel contribution.

The plausible paper contribution is the measured regime, a useful adaptive
fitting method and its justified error/cost contracts. It is not established
by proposing a representation or obtaining a few favorable fitted surfaces.

## 8. First execution batch and review boundary

The next bounded batch is **M0/M1 on the saved three-model, ten-term set**,
plus synthetic metric/reconstruction controls. It produces diagnostics and
an explicit recommendation for M2/M3, with no production code or new fits.
The pilot plan gives exact files, interfaces, tests and commands.

After that batch, review whether the evidence supports selected products,
independent rank, or localized refinement. Acquire a wide housing snapshot
before concluding that the tiny saved tensors describe rich-term savings.
Then write the bounded M2 implementation specification using the observed
geometry and the P1/P2/P3 contracts. Do not launch every branch or a new broad
dataset search while this question is unanswered.

## 9. Planning review and validation

Two bounded math/research agent reviews checked the proposed sequence,
roadmap mapping, prior test use and existing Lean scope. Their corrections
are incorporated: explicit runtime coefficient mapping, independent rich-width
scaling, the actual categorical-hour Bike model, fixed fine-space centering,
nonunique spectral truncations, and spectral-tail versus realized-error
accounting. The proof audit matched theorem-source/toolchain/manifest hashes
to the archived receipts; it did not recompile proofs for this plan.

Local checks verified eleven links between the two new plans and existing
files, parsed all five Python examples, and matched the pilot manifest to the
three saved model hashes and ten selected terms. The small metric-direction
example gives weighted squared error 4, versus 9 when whitening is deleted,
so its decision test rejects that mutation. These are planning/fixture checks,
not model fits, a production test-suite run or new numerical certificates.
