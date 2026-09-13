# Cheap discovery before a small joint interaction fit

Date: 2026-09-13. Source audited at
`03766e8d5ce5b67a2ce1091ab5ae92cabf9076c3`. This is a bounded design and
hand-derivation memo. It contains no new fits, source changes, compiled formal
proofs or measured speedups. Other agents own the current real-data trial and
the roadmap. The trial's top-eight residual-product selection among twelve
marginally shortlisted numeric predictors is a baseline, not adaptive
discovery of the interaction count.

The first experiment should discover pairs across every eligible predictor
using a small, response-independent marginal dictionary and a bounded row
sample. Existing PSST should then score the resulting shortlist on the full
training rows. Validation should choose among small joint fits, including the
additive fit. The experiment must pay for discovery, unsuccessful candidates
and all model selection. It need not compile every possible interaction.

## Count the work being done

Use the following quantities consistently.

| Quantity | Meaning |
| --- | --- |
| `d` | Number of input predictors. Also report the eligible subset and every exclusion. |
| `U = d(d-1)/2` | Potential unordered pairs before exclusions. For 384 predictors, `U = 73,536`. |
| `C_coarse` | Distinct pairs receiving the cheap discovery score. |
| `C_PSST` | Distinct pairs passed to the existing PSST machinery. |
| `C` | Distinct pairs scored by any discovery stage. Also record repeated evaluations and channel counts. |
| `s` | Interaction groups retained in the final joint model; existing fit receipts call this `M`. |
| `P = P0 + sum(p_e)` | Compiled joint coefficient width excluding the intercept, including the additive width `P0` and the retained interaction widths `p_e`. |

An all-pair scalar scan has `C_coarse = U`, even if only 64 pairs reach PSST
and only four reach the final fit. It performs all-pair discovery work, but
does not fit 73,536 interaction surfaces. A fixed cap on `C_PSST` or `s` is a
resource limit; it is not an estimate of a true powerful-interaction count.

The complete candidate cost is

\[
T_{\rm pipeline}=T_{\rm additive}+T_{\rm marginal\ preparation}
+T_{\rm coarse}+T_{\rm PSST}
+\sum_{a\in\mathcal A}T_{\rm joint\ fit,a}
+T_{\rm validation}+T_{\rm selection}+T_{\rm final\ fit}
+T_{\rm numerical\ checks}+T_{\rm outputs}.
\]

Count a reused fit once. Charge every attempted candidate and any repeated
discovery after a joint fit. A fair additive-only pipeline pays for its own
tuning and final fit under the same split and output contract. Report seconds,
the matched additive-relative ratio, phase and process peak RSS, retained
payload, fit status and actual backend dispatch. The
[budget memo](2026-09-13-cheap-interaction-budget.md) gives the wider accounting
contract. Compression of retained surfaces is the separate problem in the
[representation memo](2026-09-13-cheap-interaction-representations.md).

## What the existing PSST actually computes

The public
[`screen_interactions`](../../src/superglm/model/screening_ops.py#L543)
accepts an explicit `candidates` list. With `candidates=None`, it enumerates
eligible fitted parent pairs. It does not require either parent's fitted main
effect to pass a significance or magnitude threshold. A predictor excluded
before the additive model is built cannot enter this public candidate list.
Retaining all eligible parent specifications matters even when their fitted
main effects are weak.

The API supports spline-by-spline, spline-by-category, category-by-category,
numeric-by-category and numeric-by-numeric pairs. Unsupported parent kinds
have explicit deferrals. Selected double-penalty spline parents are rejected.
In particular, the prototype must not silently inherit main-effect variable
selection or change parent specifications to bypass this contract.

Let `a_i` be the declared observation weight, `mu_i` the additive fitted mean,
and `dot_mu_i` the inverse-link derivative at the fitted predictor. With the
same variance convention as the existing code, define

\[
g_i=a_i\dot\mu_i(y_i-\mu_i)/V(\mu_i),\qquad
w_i=a_i\dot\mu_i^2/V(\mu_i).
\tag{1}
\]

These are the
[working score](../../src/superglm/screening/_pair_moments.py#L41)
and Fisher weights used by the current screen. Gaussian identity-link scores
reduce to weighted residuals. Using raw residuals for every family is a
different score. Offsets, weight semantics and row ordering must agree with
the fitted model; subsampled calls must pass aligned weights and offsets
explicitly.

For one pair, let `N` contain its intercept and parent-main columns, and let
`B` be its candidate interaction design. Write

\[
F=W^{1/2}N,\quad Z=W^{1/2}B,\quad
\rho_i=g_i/\sqrt{w_i},\quad \Pi_F=FF^\dagger.
\]

Set `rho_i = 0` on zero-weight rows, where the score must also be zero.
On the supported nuisance space the exact quantities are

\[
U=Z^\top(I-\Pi_F)\rho,\qquad
V=Z^\top(I-\Pi_F)Z,\qquad
T_\lambda=U^\top(V+\lambda S)^{-1}U.
\tag{2}
\]

All inverses here mean identified-space inverses. The source obtains `U` and
`V` through trailing blocks of a
[weighted design factor](../../src/superglm/screening/_pair_factor.py#L1),
with rank-aware nuisance handling. It does not compute a difference of two
large Grams to profile the nuisance block. This matters on starved cells and
nearly absorbed interactions.

The nuisance span in (2) is the candidate pair's parents, not the full fitted
additive design. The baseline residual reflects all fitted additive terms,
but that fact does not turn (2) into a fully profiled joint-model update.
Correlated predictors and penalized parent fits make this distinction material.

The
[score ladder](../../src/superglm/screening/_score_stat.py#L1)
selects penalties for EDF budgets and returns the largest standardized score.
For shrinkage eigenvalues `a_j(lambda)`,

\[
z_\lambda=
\frac{T_\lambda/\phi-\sum_j a_j(\lambda)}
{\sqrt{2\sum_j a_j(\lambda)^2}}.
\tag{3}
\]

The [reference-variance derivation](2026-09-psst-reference-variance.md) assumes
fixed geometry and `U ~ N(0, phi V)`. It does not calibrate the maximum over
pairs and EDF rungs after fitting the baseline. PSST is a ranking tool.
Its numerical rank checks and cost refusals do not establish predictive
usefulness. A `NaN` refusal is unresolved evidence, not a zero score.

For gridded margins the code makes one fused `O(n)` score/weight cell pass per
pair. Marginal preparation is cached across that call. Dense design-factor
construction and its decomposition then depend on support and candidate
width, with cubic width work for the latter. The numeric-by-numeric route
already provides a small exact factor for `[1, x_j, x_k, x_j*x_k, response]`.
Binning can change the spline candidate geometry and is flagged `approx`;
it is not a certified approximation to every unbinned score.

## A cheap broad scan and its limits

Freeze a small dictionary `psi_j1,...,psi_jh` from training predictors, without
examining their marginal relationship to `y`. Center and scale it using
training data. A useful initial dictionary has the centered linear direction
and the lowest positive-penalty mode of the fitted spline margin. Obtain the
latter from the existing compatible marginal basis and penalty ingredients;
do not silently substitute a different spline penalty. Unsupported predictors
need a declared typed dictionary or an explicit exclusion.

For `m` sampled training rows and a chosen pair of dictionary channels, compute

\[
\widehat u_{jk,ab}=\frac nm\sum_{i\in I}
g_i\psi_{ja}(x_{ij})\psi_{kb}(x_{ik}),\qquad
\widehat v_{jk,ab}=\frac nm\sum_{i\in I}
w_i\psi_{ja}(x_{ij})^2\psi_{kb}(x_{ik})^2.
\tag{4}
\]

Use `uhat^2/vhat` only where the denominator is numerically resolved, and use
its magnitude to propose candidates. This is an unprofiled one-direction
working-quadratic score. Even on all rows it omits the projection in (2).
The scaling factor is useful when comparing different row budgets; it is
common within a single scan. A supplied dictionary normalization does not
remove this projection obligation.

Each `a,b` channel admits blocked matrix products

\[
\Psi_a^\top\operatorname{diag}(g)\Psi_b,
\qquad
(\Psi_a^{\circ2})^\top W\Psi_b^{\circ2}.
\tag{5}
\]

They visit every predictor pair in `O(m d^2 h^2)` conventional arithmetic.
They store `O(m d h)` values, plus a `d`-by-`d` score block or smaller tiles and
a bounded shortlist. No `n`-by-`U` tensor design is necessary. At `d=384` and
`m=4096`, one unordered channel visits 301,203,456 pair-row contributions.
Two marginal channels have four channel combinations. BLAS execution and
symmetry affect constants, so this is an operation count, not a time estimate.
Using all `n` rows instead costs `O(n d^2 h^2)` and must be timed as such.

Keep a union of high-scoring pairs from each channel as well as an aggregate
ranking. This gives a nonlinear channel some candidate budget even when a
linear channel produces larger scores. Channel quotas are design choices,
not calibrated statistical thresholds. A next resolution can add more
compatible spline modes or training-frozen local bin contrasts; every extra
cross-channel scan has a cost. Searching high resolution only on the existing
shortlist cannot rescue a pair whose low-resolution witnesses were all weak.

A pair-specific learned rank-one witness also has a precise limitation. For
fixed, profiled geometry, optimizing over a restricted product direction can
only attain a gain at most the full candidate's gain. Searching directions
may cost additional passes or nonconvex iterations. A weak rank-one witness
is not an upper bound on an arbitrary interaction, and cannot safely rule it
out. Taking a leading singular vector of the raw score matrix is not the
solution under general empirical curvature and nuisance projection.

This row sample is the proposed first sketch. Random feature aggregation or
heavy-hitter sketches might avoid enumerating every pair, but this memo
establishes no recovery theorem, sketch dimension or implementation for them.
Arbitrary pair scores do not acquire a cheap recovery guarantee merely
because the desired final model is sparse.

## Conditional error statements

The following are hand derivations. They concern stated finite-dimensional
quantities and are not implemented certificates in the public API.

For a full-row scalar candidate, let
`t_tilde = (I-Pi_F) W^(1/2) t`, `u = t_tilde' rho` and
`v = ||t_tilde||^2`. If computed values have justified absolute error bounds
`|uhat-u| <= e_u`, `|vhat-v| <= e_v`, and `vhat > e_v`, then

\[
\frac{\max(0,|\widehat u|-e_u)^2}{\widehat v+e_v}
\ \leq\ \frac{u^2}{v}\ \leq\
\frac{(|\widehat u|+e_u)^2}{\widehat v-e_v}.
\tag{6}
\]

For a known positive scalar penalty, add it to both denominators. A decision
whose intervals overlap requires more evidence. If the denominator cannot
be resolved, refine the arithmetic or refuse the decision. Do not impose a
fixed small-variance cutoff and call it a numerical theorem.

For the raw numerator in (4), a conventional floating-point dot-product
analysis bounds accumulation error by a dimension-dependent `gamma_l` times
the absolute term sum. Here `gamma_l = l*u/(1-l*u)`, with dtype roundoff `u`
and `l` covering the actual multiplication and reduction schedule. The bound
requires no overflow or unaccounted underflow. Marginal-evaluation, fitted
score, scaling and nuisance-projection errors must also be enclosed; a dot
product bound alone does not enclose (2). Cancellation can make the resulting
relative error large. No current PSST output is claimed to contain the
explicit interval (6).

There is a separate conditional sampling calculation. Freeze the full-data
baseline and one channel's finite population
`b_i = g_i*psi_ja(x_ij)*psi_kb(x_ik)`. Under simple random sampling without
replacement,

\[
E(\widehat u\mid b)=u,\qquad
\operatorname{Var}(\widehat u\mid b)
=\frac{n^2(1-m/n)}m S_b^2,
\quad
S_b^2=\frac1{n-1}\sum_i(b_i-\bar b)^2.
\tag{7}
\]

The identity follows by the inclusion probabilities for one and two sampled
rows. Chebyshev's inequality gives a conditional bound by dividing this
variance by the squared error threshold; a union bound over all evaluated
channels requires no independence. Plugging sample variances into (7)
does not supply a valid population variance bound.

For an explicit conservative construction, let `L` be the total pair-channel
count, choose `0 < delta < 1`, and assume training-frozen bounds
`|b_i,l| <= B_l` for score summands and `0 <= c_i,l <= D_l` for curvature
summands. These can follow from full-training maxima of `|g|`, `w` and each
dictionary column, without constructing interaction columns. For `n > 1`,
both finite-population variances are at most `n/(n-1)` times their respective
squared bound. Thus define

\[
e_{u,l}=nB_l\sqrt{\frac{n}{n-1}\frac{1-m/n}{m}
\frac{2L}{\delta}},\qquad
e_{v,l}=nD_l\sqrt{\frac{n}{n-1}\frac{1-m/n}{m}
\frac{2L}{\delta}}.
\tag{7a}
\]

Chebyshev and a union bound give simultaneous numerator and denominator
coverage at least `1-delta`, conditional on the frozen full-training values.
The proof uses `sum((b_i-mean(b))^2) <= sum(b_i^2) <= n B_l^2`, and the same
bound for `c`. No channel independence or independence from the fitted
baseline is needed for this conditional sampling statement. Arithmetic
enclosures must be added for a computed implementation.

Apply (6) to the raw full-row proxy with these errors whenever its lower
denominator bound is positive. Use an unresolved upper bound otherwise.
For the pair aggregate `R_e = max_ab u_e,ab^2/v_e,ab`, take the maximum of
the channel lower bounds and the maximum of their upper bounds. If `tau_K`
is the `K`th largest pair lower bound, a pair with upper bound strictly below
`tau_K` cannot enter the top `K` of this full-row proxy on the simultaneous
coverage event. Retain ties and refine unresolved pairs on all rows. This
protects the ranking of the specified proxy only. It does not protect PSST
rankings, ranks of design matrices, or selection by predictive usefulness.

The factor `sqrt(L/delta)` is usually severe for tens of thousands of pairs.
Bounds from global maxima are worse with rare large weights or localized
signals, and denominator intervals may include zero. The protected shortlist
may contain every pair. A hard cap that drops unresolved candidates loses
this coverage claim. More sophisticated concentration could improve this
bound under additional assumptions, but is not established here. A measured
unprotected sampled proposal remains a legitimate heuristic if labelled as
such. These sampling facts are not generalization statements or
sure-screening results for rich interactions.

An exact full-space omission bound explains the additional evidence needed
for safe removal. At a fixed penalty, partition an identified positive
definite candidate system into inspected and omitted directions:

\[
H=\begin{bmatrix}A&B\\B^\top&D\end{bmatrix},\quad
U=\begin{bmatrix}u\\v\end{bmatrix},\quad
Q=D-B^\top A^{-1}B,\quad r=v-B^\top A^{-1}u.
\]

Block elimination gives

\[
T=u^\top A^{-1}u+r^\top Q^{-1}r.
\tag{8}
\]

If `Q >= delta I` for a justified `delta > 0`, the omitted contribution is at
most `||r||^2/delta`. This follows by the inverse order bound. Numerical use
also needs enclosures for the residual, solves and coercivity. Obtaining `r`
can require all omitted rich products. Weak penalties or unidentified null
directions can make the bound unavailable or useless. Low-resolution scores
alone supply neither `r` nor `delta`. Equation (8) also does not preserve
rankings under the changing EDF normalization in (3).

For fixed geometry, `T/2` is the optimum gain of the corresponding profiled
working quadratic. In Gaussian regression this is an exact quadratic
statement for the declared nuisance and penalty problem. PSST's pair-local
nuisance problem still differs from a full additive-plus-interactions joint
fit. A changing GLM working problem or re-estimated smoothing adds further
differences. None of (6)-(8) bounds held-out predictive gain.

## One practical prototype

### Usefulness and basis resolution are separate decisions

The user's further requirement is to retain predictive structure rather than
noise, and to choose how much basis resolution each retained interaction
needs. PSST does not settle either decision. Its winning `edf0` maximizes a
screening statistic over smoothing levels in a supplied candidate space.
The [public contract](../../src/superglm/model/screening_ops.py#L543) describes
it as screening complexity. It is neither an optimal knot count nor evidence
that those basis functions will improve held-out predictions.

Distinguish the represented space, its coefficient dimension, and its fitted
effective degrees of freedom. A wide basis with strong smoothing can have low
EDF. A narrow basis cannot represent an omitted fine-scale direction, however
its smoothing is chosen. An EDF rung near the top of the screen can motivate
a richer probe, but is not a basis-adequacy certificate. Low EDF is not a
certificate of adequacy either.

The bounded search therefore has two axes: the selected pair set `E` and a
resolution `h_e` for each selected pair. Include the additive model, coarse
interaction models and a small number of training-generated refinements.
Re-estimate smoothing and every coefficient jointly for each admitted model.
Compare each addition or refinement with its simpler parent using paired
held-out loss, not training fit improvement or the PSST score. Prefer a
simpler model when the evidence does not resolve a useful gain under the
declared selection rule. Charge all attempted refinements.

A refinement probe must condition on the coarse interaction already fitted
and the other selected terms. Re-scoring the original pair can merely recover
its existing coarse signal. The pair-parent projection in current PSST does
not supply that full adjustment. Equation (8) describes the relevant fixed-
quadratic block correction, including cross-penalty terms where present;
its construction and numerical certification remain implementation work.

Initially, ordinary lower-knot tensors are useful controls. They are not
automatically nested and their normalized penalties can change. The proposed
shared-basis dictionaries instead freeze a rich space and pull back its
penalty, permitting same-objective numerical comparisons. That certification
question remains distinct from selecting a predictive resolution after
re-estimating smoothing.

Coarse-to-fine selection needs an exploration budget. A pair can have weak
coarse scores and a strong localized or higher-frequency interaction. Testing
richer directions only for coarse survivors would make that blind spot
permanent. Predeclare some richer-direction probes outside the coarse
shortlist, record their cost, and measure missed-signal rates on controlled
alternatives. No coverage guarantee follows merely from reserving that budget.

The validation search must also be bounded. Generate the model menu using
training data only, then use validation for selection and untouched outer
test data for the final procedure. Repeatedly designing new bases after
looking at validation loss can overfit that validation set. Repeated or
grouped folds assess stability when appropriate to the sampling unit; a
one-standard-error rule is a model-selection heuristic, not a calibrated
probability that an interaction is real. Chronological tests can still reveal
distribution shift, as the current bike pilot did.

One conditional statistical bound makes the required assumptions explicit.
Freeze `K` candidate models using training data only, with an additive
comparator. On `m` independent validation observations from the target
distribution, let the paired improvement for candidate `a` be
`D_ai = loss(additive, i) - loss(candidate_a, i)`. Assume a known population
bound `-B <= D_ai <= B`, with `B > 0`. Applying the bounded-variable
inequality in [Hoeffding, 1963, Theorem 2](https://doi.org/10.1080/01621459.1963.10500830)
and a union bound gives, simultaneously for every candidate, with probability
at least `1-delta`,

\[
E[D_a\mid\mathrm{training}]
\geq\overline D_a-B\sqrt{2\log(K/\delta)/m}.
\tag{9}
\]

For one candidate the upper-tail probability is at most
`exp(-m*t^2/(2*B^2))`; substituting the displayed threshold gives `delta/K`,
and summing over candidates proves (9). Dependence between candidates on the
same validation rows is allowed. This is a conditional hand derivation, not
a new formal proof or an implemented acceptance rule. A positive lower bound
would support positive expected predictive gain for the selected procedure
under these assumptions; it would not identify a causal or unique true edge.

The bound can be too loose to be useful. Current raw squared-error and Poisson
losses have no such population bound, and dependent time or subject rows are
not independent validation observations. A maximum observed loss is not a
valid substitute for `B`. Clipping the loss changes the target criterion.
An adaptively invented model menu cannot use just the number visited as `K`
in this argument. Sharper guarantees require stated tail/dependence and
selection assumptions. Numerical error bounds for fitted predictions and
loss computation must be accounted for separately; none of the current pilot
results carries the statistical certificate in (9).

### Initial bounded implementation

Use a standalone research experiment with the existing additive model,
working-score functions, marginal ingredients and public PSST call.

1. Freeze train, validation and untouched test splits using the dataset's
   appropriate entity or time unit. Fit the additive model with all eligible
   parent specifications. Construct knots, marginal dictionaries and category
   rules from training predictors only. Record all exclusions.
2. Compare two proposal arms with the same `h = 2` compatible marginal
   directions. The reference contracts (4)-(5) over all training rows and is
   the exact full-row proxy in real arithmetic. The sampled arm uses
   `m = min(n, 4096)` training rows with a recorded seed. Evaluate all eligible
   numeric pairs in both arms. The full-row proxy is still unprofiled and has
   the dictionary's approximation limits. This first experiment deliberately
   addresses numeric discovery;
   report its coverage separately on mixed datasets. Do not shortlist its
   predictors by marginal response association. Keep unresolved denominator
   cases in a separate recorded queue for full-row checks.
3. Construct nested candidate sets of at most 32, 64 and 128 distinct pairs
   using the channel union. Score the largest set once on all training rows
   with `model.screen_interactions(..., candidates=pairs)`. Reuse its rows for
   the smaller sets. Preserve current EDF defaults, approximation flags,
   numerical refusals, tolerances and weight semantics. The public API does
   not expose a persistent cache across separate screening calls.
4. For each candidate budget, fit a small predeclared grid of joint models,
   for example PSST prefixes of sizes 0, 2, 4, 8 and 16 where available. Fit
   additive and selected interaction coefficients together. Deduplicate equal
   models across budgets. Choose the final budget and model by validation
   loss with a declared simplicity tie rule and an explicit runtime limit.
   The result is a validation-selected `s`, bounded by the experiment, not a
   recovered true count. Evaluate the selected procedure once on the test set.
5. Compare with the current twelve-feature/top-eight baseline, the additive
   baseline, and direct all-pair PSST on a deliberately small predictor count
   where it is affordable. The latter is a diagnostic for discovery misses;
   its answer must not influence the cheap arm. On synthetic data also report
   known-signal pair recovery, while treating predictive loss as the real-data
   outcome. Log `d, U, C_coarse, C_PSST, s, P`, all attempts and total cost.

This prototype asks whether broad cheap scores preserve useful pairs under a
small refinement budget. An `h=4` or larger-row comparison is justified if
its miss diagnostics show a need. It is another measured arm with its own
budget. Success on one seed or one dataset does not establish a screening
guarantee.

A regularized joint fit is an alternative to the prefix grid. Group lasso or
another declared group penalty can select `s` among a somewhat wider PSST
shortlist, with tuning chosen on validation data. Verify that the selected
interaction classes support the intended penalty and solver. This changes
the fitting objective and does not recover candidates excluded upstream.
Ordinary spline smoothness penalties alone need not set an interaction's
unpenalized directions to zero. An EDF cutoff is not automatically a group
selection rule.

The numeric experiment does not cover the corpus's mixed interactions. The
bike pilot illustrates the gap: hour and month remained categorical, while
the numeric-only screen could examine just the six pairs among four weather
predictors. It could not propose hour-by-weather or workday-by-hour. Preserving
the category levels in the final model does not repair a proposal rule that
never pairs those variables.

A bounded mixed extension can use all `L_j-1` centered categorical contrasts
for small factors, for example factors with at most 32 retained training
levels, if the declared work budget admits their total width. Whiten supported
contrast directions with rank-aware factor algebra; rare or zero-weight
levels need explicit treatment and recorded deferral. This avoids assigning
an arbitrary numeric order to unordered categories. It still admits
dimension and work refusals. With variable dictionary widths `h_j`, replace
`d*h` by `H = sum_j h_j`; an all-pair contraction costs `O(m H^2)` before
within-predictor blocks are omitted. Full categorical contrasts have no
direction truncation within their supported categorical span, but score
aggregation, finite samples and the shortlist cap can still miss useful pairs.

For predictors with declared periodic semantics, such as hour of day, two
training-centered sine/cosine columns at the declared period provide a
bounded proposal dictionary. They require no response-based main-effect gate.
Additional harmonics cost additional channels; the first harmonic can miss
localized or higher-frequency signals. A periodic proposal may nominate a
categorical hour parent for the existing mixed PSST and its declared refit
target. It does not make that target a periodic spline model. Use all training
rows for PSST and every joint fit in either extension. Factors that exceed
the mixed dictionary budget remain explicitly deferred, not declared screened.

## Required stress cases and unresolved claims

| Case | What it tests |
| --- | --- |
| Independent pure interaction with zero main effects | Use `y = f(x_j)g(x_k)+noise` with centered factors and vary the pair location. Main-effect screening can miss it completely; the proposed broad scan has no such eligibility gate. |
| A pure interaction orthogonal to both cheap dictionaries | Demonstrates that two-mode scores can be zero despite useful rich interaction signal. The shortlist is allowed to fail; its failure must be visible. |
| Correlated and nearly duplicate predictors | Tests additive leakage, pair-local nuisance limitations, proxy edges and unstable individual coefficients. Evaluate aggregate predictions and identification/refusal evidence separately from exact pair labels. |
| Localized or sign-changing interaction | Tests row-sample omission and cancellation across coarse modes. If `r` of `n` rows carry a localized signal, the chance a uniform sample misses all of them is `choose(n-r,m)/choose(n,m)` when feasible. Higher modes only on survivors cannot repair this miss. |
| Additive null and many irrelevant predictors | Tests maxima over channels, pairs and EDF budgets, and whether validation selects `s=0`. A fixed positive prefix cap cannot itself establish a nonzero signal count. |
| Rare categories, imbalanced weights and sparse counts | Tests unreliable small denominators, row-sample coverage and family-correct scoring. Any stratified sampling arm needs its own inclusion-weight formulas; equation (7) no longer applies unchanged. |
| Several correlated true edges, mixed signs and unequal strengths | Tests whether individually strong pairs remain useful jointly and whether a weak individually scored pair becomes useful after another edge is fitted. Charge any repeated residual-screening rounds. |

No theorem here establishes sure screening, a calibrated null cutoff, a
population-risk guarantee, recovery of the number of powerful interactions,
or a constant additive-relative runtime. Randomized discovery has no asserted
probability guarantee beyond the conditional calculation (7) under its stated
sampling design. The existing
[PSST detection study](2026-09-psst-detection-study.md) is empirical evidence
for shortlist evaluation, not a substitute for these missing guarantees.

Numerically safe scoring can establish that a specified score or solve was
resolved accurately, or that a decision needs refinement. It cannot establish
that an unexamined interaction is predictively useless. The practical target
is a measured time/RSS/held-out-loss frontier for the entire discovery and
joint-fit procedure, with numerical error and statistical selection error
reported separately.
