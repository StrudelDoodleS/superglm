# An additive-normalized budget for cheap interactions

Date: 2026-09-13. Audited research head:
**03766e8d5ce5b67a2ce1091ab5ae92cabf9076c3**. Production source is unchanged
from the preceding many-interaction analysis. This note owns no new solver,
fit or formal proof. Its mathematical statements are labelled hand derivations
or conditional cost models. The
[many-interaction analysis](2026-09-13-many-interaction-scaling-analysis.md)
contains the fuller source audit.

The user's objective is useful interactions that remain cheap relative to an
additive fit, with roughly ten times additive time as an exploratory scale,
not a fixed acceptance threshold. The immediate opportunity is concrete:
unchanged Gaussian tensor cross-products are currently recomputed across
coefficient-solver invocations. Model compression is a separate route with
different statistical obligations.

The later user clarification makes discovery central: cheaply identify a
small, adaptively selected set among many possible interactions. In this note,
retained `M` is distinct from the potential pair universe. Candidate screening
and model-size selection must be charged even when the final model is small.
The [discovery analysis](2026-09-13-cheap-interaction-discovery.md) gives the
next experiment; exhaustive all-pairs tensor fitting is not the workload.

## Define the baseline and the allowance

Let \(T_0^{\rm fix}\) be the complete additive fit with fixed smoothing and
\(T_0^{\rm opt}\) the complete additive fit with automatic smoothing. Compare
each interaction fit with its corresponding baseline:

\[
R_T^a(M)=\frac{T^a(M)}{T_0^a},\qquad
\Delta T^a(M)=T^a(M)-T_0^a,\qquad a\in\{\mathrm{fix},\mathrm{opt}\}.
\]

**Hand derivation.** A guide \(T(M)\leq cT_0\) is equivalent to
\(\Delta T(M)\leq(c-1)T_0\). For \(M>0\), its average incremental allowance is

\[
\frac{\Delta T(M)}M\leq\frac{(c-1)T_0}{M}.
\]

At \(c=10\), that is \(9T_0/M\) per retained interaction. It is an allowance,
not a claim that each interaction costs the same. The marginal difference
\(T(M)-T(M-1)\) also includes changed optimizer iterations, reuse, dispatch and
model geometry; it can even be negative. Do not estimate a universal marginal
cost from two endpoints.

Record \(M_{\rm examined}\), retained \(M\), effective interaction widths
\(p_g=k_{g,u}k_{g,v}\), total
\(P=P_0+\sum_g p_g\), smoothing dimension \(q\), and any shared rank \(r\).
Examining many candidates and retaining a few is different from fitting many
independent rich surfaces; the candidate work belongs in the budget.

For each arm, decompose complete time into non-overlapping phases:

\[
T(M)=T_{\rm prepare}+T_{\rm search}
 +T_{\rm build}+T_{\rm all\ coefficient\ fits}
 +T_{\rm smoothing\ overhead}+T_{\rm certify}
 +T_{\rm finalize}+T_{\rm requested}.
\]

If finalization includes a coefficient refit, assign that refit once. Charge
failed candidates, line searches, rank/resolution selection, tuning and
requested inference. Public-fit timings and wider pipeline timings should
both be labelled: the current fit clock excludes imports, fixture generation
and subsequent prediction/export. A future selector must include those of its
additional preparation/prediction passes that it requires to choose a model.

**Conditional phase model.** If a fraction \(f\) of additive time is shared
setup and the remaining phase grows by a factor \(a\), then
\(T(M)/T_0=f+(1-f)a+T_{\rm new}/T_0\). A small ratio on a setup-dominated
fixture does not establish a small large-problem ratio. Conversely, a very
fast additive baseline makes a modest absolute interaction cost look large.
Always report seconds alongside ratios.

## What the fresh baseline already says

The parent-owned
[matched probe](2026-09-13-cheap-interaction-additive-baseline.md) and
[receipt](2026-09-13-cheap-interaction-measurements.json) cover
\(M=0,8,16,28\), fixed smoothing and REML, with two serial repetitions in
opposite orders. All use the same 2,048 rows,
eight parent splines of effective width five, 64 bins and width-25 tensors.
All fits converge with Gram dispatch. The endpoint medians reported by that
probe are:

| Mode | Additive \(T_0\), seconds | 28 interactions, seconds | \(T(28)/T_0\) |
| --- | ---: | ---: | ---: |
| Fixed smoothing | 0.177184 | 0.733937 | 4.14 |
| Automatic smoothing | 0.366325 | 4.163770 | 11.37 |

These are unchanged-source observations, not an optimization speedup.
Reaching the \(10T_0\) guide in the automatic arm would mean about 3.663 seconds,
or 12.0% less complete-fit time than this matched median. That is arithmetic,
not a forecast for the reuse proposal. Do not mix an older interaction timing
with this new additive denominator.

The REML test MSE is 0.740723 for additive and 0.110462 at 28 interactions,
against synthetic noise variance 0.09. The response law contains all 28 edge
signals, each a separable rank-one function. It is a useful demonstration that
interactions can buy prediction quality, but cannot establish performance on
arbitrary rich surfaces or a general accuracy guarantee. The parent owns the
complete measurement receipt, repetition details and retained-memory census.

Its coarse-interaction controls retain the same additive parents while changing
only interaction resolution. Those are different statistical models. Better
held-out loss for a coarse fit can be useful regularization; it does not certify
that fit as an accurate numerical approximation to the richer fitted problem.
All completed, refused and failed harness outcomes belong in the report.

## The useful conditional regimes

No universal constant such as ten can cover arbitrary \(M\), width, conditioning
and output requirements. Even at fixed \(n,P_0\), an explicit vector of
\(\sum_g p_g\) arbitrary interaction coefficients has unbounded output size as
those dimensions increase. An explicit full covariance requires
\(\Omega(P^2)\) entries. These are output-size arguments, not a cubic fitting
lower bound. The relevant question is which stated regimes have a favorable
accuracy/time/memory frontier.

**Local actions with shared marginals.** Suppose there are \(d\) additive
features, at most \(s\) local basis entries per margin, and all constraints/maps
have compatible cheap actions. One additive data action visits \(O(nds)\)
entries; \(M\) separate product interactions add \(O(nMs^2)\). In this
data-action-dominated model, the ratio is of order

\[
1+\frac{Ms}{d}.
\]

This follows by dividing the work counts; it is not the measured fit ratio.
Sharing parent evaluation removes duplicated marginal preparation, but
arbitrary edge coefficients still need edge contractions. Complete all-pairs
interactions have \(M=d(d-1)/2\), so the ratio grows with \(d\) under this
particular representation. Fixed degree/local overlap helps absolute work;
increasing knot count need not increase row overlap, but it can still increase
penalty, factor, certificate and output costs. Current dense maps and cardinal
tensor tables do not automatically attain the local-action bound.

**Few additional directions with an existing factor.** For fixed weights,
unchanged base penalty and compatible coordinates, adding one group of width
\(p\) gives a coupled system

\[
\begin{bmatrix}H_0&C\\C^\top&D\end{bmatrix}
\begin{bmatrix}\beta_0\\\gamma\end{bmatrix}
=\begin{bmatrix}b_0\\b_g\end{bmatrix}.
\]

**Hand derivation.** With \(V=H_0^{-1}C\) and
\(Q=D-C^\top V\), solve
\(Q\gamma=b_g-C^\top H_0^{-1}b_0\), then
\(\beta_0=H_0^{-1}b_0-V\gamma\). This is the exact joint solve when the
assumed factors are nonsingular. Given a conventional dense factor of \(H_0\),
the additional algebra costs
\(O(P_0^2p+P_0p^2+p^3)\), plus constructing the cross/diagonal blocks and
validating the numerical solve. The new raw row products cost
\(O(\sum_i a_{i0}a_{ig}+\sum_i a_{ig}^2)\) under local evaluation, before maps.

Fitting the new group to additive residuals while keeping the additive
coefficients frozen omits the Schur correction and back reaction; that is a
different fitting procedure. Re-estimating base smoothing parameters can also
invalidate the old factor. Repeating dense block updates as \(P\) grows does
not remove the overall cubic conventional factor work. This route is useful
for genuinely small additions, not a promise for an arbitrary long sequence.

**Operator solves.** The coupled identity
\(Hv=X^\top(WXv)+Sv\) avoids enumerating interaction pairs. Under the local
representation assumptions its action costs
\(O(n+P+z+C_S+t)\), with row overlap \(z\), structured penalty work \(C_S\)
and coordinate-map work \(t\). A cheap action is useful when conditioning,
preconditioner setup, coefficient iterations, smoothing traces/log determinants
and requested uncertainty also remain affordable. The baseline may already
reuse a fixed Gaussian Gram; compare its one-time assembly and subsequent
factors with all operator passes, not with one action.

**A shared factorization can make all pairs cheap per pass.** For functions
\(h_{v\ell}\) represented in centered parent bases, consider the restricted model

\[
f_{\rm int}(x)=\frac12\sum_{\ell=1}^{r}\sigma_\ell
\left[
 \left(\sum_v h_{v\ell}(x_v)\right)^2
 -\sum_v h_{v\ell}(x_v)^2
\right].
\]

**Hand derivation.** Expanding the square leaves
\(\sum_{\ell}\sigma_\ell\sum_{u<v}h_{u\ell}(x_u)h_{v\ell}(x_v)\).
Thus all pairs can be evaluated without enumerating edges. With bounded \(r\),
one value/gradient pass costs \(O(nr\sum_v s_v)\), up to the scalar
accumulation work; parent-basis coefficients number \(O(r\sum_v k_v)\).
Relative to comparable additive passes, this is \(O(r)\). Arbitrary edge
masks require their own cost; the square identity is for the complete pair sum.

This is a statistical restriction. Rank one on each edge does not imply one
global shared factor: four scalar margins with
\(w_{12}=w_{13}=w_{14}=w_{23}=w_{24}=1,w_{34}=2\) violate the rank-one identity
\(w_{12}w_{34}=w_{13}w_{24}\). Conversely, the rank of a coefficient matrix
whose diagonal was set to zero is not alone the required shared rank, since
the diagonal completion is free when self terms are subtracted.

Learning the factors changes the linear convex coefficient problem into a
nonlinear factor problem. Iterations, conditioning, initialization/restarts,
penalty choice and selection of \(r\) must all be charged. Low effective rank
of a rich fitted interaction likewise does not reduce its cost unless the
representation/solver uses it and certifies the discarded contribution.
Neither observation establishes \(T(M)=O(rT_0)\) for complete automatic fits.

**Control smoothing dimension deliberately.** Independent anisotropic tensors
usually add two smoothing parameters each. Fixed smoothing avoids that outer
work; isotropic, shared or grouped smoothing can reduce \(q\), but changes
adaptivity. Rank selection and selecting those grouping rules are also model
selection. Existing full outer-Hessian work includes pair traces, coefficient
response solves and a \(q\times q\) system. A bounded shared rank with
unbounded tuning trials is not a bounded-cost fitting procedure.

## First testable hypothesis: keep invariant Gaussian geometry

The preceding profile found nine centered-system builds for 28 small tensors.
Each coefficient-solver invocation starts a new
[constant-weight cache](../../src/superglm/solvers/irls_direct.py#L1145).
Unprojected multi-penalty tensor groups remain unchanged at
[dm_builder.py:1225](../../src/superglm/dm_builder.py#L1225), whereas ordinary
SSP maps change with lambda in the branch starting at line 1258. These source
facts explain a narrow opportunity; carrying the entire transformed Gram
unchanged would be incorrect.

Let \(F\) be the number of assemblies, \(C_{\rm TT}\) the work for reusable
tensor-by-tensor products in one assembly, \(C_{\rm admit}\) the complete
comparison/evidence cost, and \(C_{\rm manage}\) the added copying/management
cost. A first cost hypothesis is

\[
T_{\rm old}-T_{\rm reuse}
\approx(F-1)C_{\rm TT}-C_{\rm admit}-C_{\rm manage}.
\]

This is an accounting hypothesis, not a speedup bound: cache behavior,
concurrency and downstream execution can change. Existing profile clocks
overlap and contain instrumentation overhead; do not substitute their
cumulative seconds into this expression to predict uninstrumented savings.

The useful pilot has three explicit questions:

| Hypothesis | Observation required |
| --- | --- |
| Fit-local ownership avoids repeated invariant tensor products | Their build count falls across bootstrap, candidate and finalization calls, while changed main maps and penalties still produce the correct new targets. |
| Reuse beats its admission/storage cost | Matched complete-fit seconds fall; fit-phase/whole-process peak and retained bytes are reported, including old/new owners and snapshots. |
| A compatible contraction plan avoids per-bin replanning | Plan construction count falls from the identified per-bin pattern; numerical targets and arithmetic/error requirements still pass. |

Admission must bind ordered basis values, dtype, row/bin maps, weights,
coefficient placement, centering and numerical-policy evidence. Positive
lambda changes need fresh penalties/Hessians and valid factor evidence;
changed weights, basis or maps cannot inherit incompatible products. Preserve
zero-face and refusal behavior. A cache shared only by identity cannot establish
safety under mutation. Per-matrix limits also do not bound the sum of cached
histograms; retain only the state whose measured reuse justifies its cost.

Start with the matched \(M=0,8,28\) small-tensor cases and an existing wide-term
control, within their established run budgets. The implementation and baseline
must use the same current source for unaffected behavior, and both additive
denominators must be remeasured after a change. A general raw-moment improvement
could speed additive fits too, altering the normalized ratio even when both
absolute times improve. Production changes require the repository's focused
numerical and separate performance/dispatch regressions; none is made here.

The next hypothesis depends on this result. If pair assembly remains dominant,
test a bounded local/shared-factor representation against a declared rich
reference. If coefficient factors or smoothing dominate, first establish the
operator/preconditioner and \(q\)-control budgets. This is a choice from the
complete profile, not authorization for a new collection of unrelated solvers.

## Judge an accuracy/time/memory frontier

### Goal clarification and interpretability, 2026-09-14

The user revisited whether making interactions cheap was an ill-defined or
unpromising objective. The useful target remains conditional on workload,
accuracy and the requested model outputs. Distinguish the cost of one known
interaction, discovering useful pairs, and jointly fitting many coupled
terms. The [broad trial](2026-09-14-broad-interaction-trials.md) provides
useful reference groups but does not resolve all three costs.

For a single pair with b retained basis functions per margin, a full tensor
stores b² coefficients. A factorization C=UVᵀ with rank at most r stores 2br
factor entries. This elementary storage count is not an identifiable-parameter
count, fit-time bound or approximation guarantee. An arbitrary C need not
admit a useful low-rank approximation. Function-error certification must
account for the basis geometry and observation measure. Rügamer's published
AFM construction supplies evidence for scalable factorized spline models,
with complexity dependent on the factor count. It does not establish the
cost of SuperGLM's complete REML and inference contract.
[Primary paper](https://proceedings.mlr.press/v238/ruegamer24a.html)

An absolute claim that GBM interactions cannot be separated is incorrect.
Lengerich et al. give an exact functional-ANOVA purification algorithm for
piecewise-constant functions, including tree models. Its interpretation
requires a specified distribution and identification convention. The same
need for explicit conventions applies to smooth interaction terms.
[Primary paper](https://proceedings.mlr.press/v108/lengerich20a.html)

Boosting also supports deliberately additive/pairwise models; EBM is one
example. SuperGLM's intended value should be evaluated through its explicit
smooth terms, supported model controls, numerical contracts and measured
cost, with such structured competitors included in comparisons.
[EBM documentation](https://interpret.ml/docs/ebm.html)

The immediate representation question is the cost needed to preserve a
known useful interaction to a declared error/accuracy target. Retain the
separate numerical and statistical tolerances below. A target such as ten
additive fit times remains a workload-dependent aspiration, not a universal
promise or a mathematical limit. No new fits or formal proofs accompany
this clarification.

For each candidate \(a\), report
\((L_a,T_a/T_0,\mathrm{RSS}_a,\mathrm{RSS}_a/\mathrm{RSS}_0,B_a)\),
where \(L_a\) is held-out loss and \(B_a\) retained model payload. Include
convergence/certification status, examined and retained term counts, widths,
rank, smoothing dimension and absolute seconds. For a loss tolerance
\(\varepsilon_{\rm stat}\) relative to a useful rich reference, a conditional
acceptance set is

\[
\mathcal A=
\{a:\ L_a\leq L_{\rm rich}+\varepsilon_{\rm stat},\
T_a/T_0\leq c,\
\mathrm{RSS}_a\leq B_{\rm peak},\
\text{required numerical contracts pass}\}.
\]

Initially, vary \(c\) around the exploratory scale and display the tradeoff;
the user has not prescribed a hard ten-times gate. Choose the loss tolerance
and memory budget before the final untouched evaluation, and quantify
held-out sampling variation using the dataset's appropriate sampling unit.
An observed test-loss inequality is not a population-risk certificate.
If the rich model
does not improve on additive, do not use its loss difference as a stable
normalizing denominator. Separate equivalent-system numerical error from
statistical error introduced by low rank, tied smoothing, screening or reduced
resolution.

Fair comparisons retain the same observations, parent function spaces,
family/link, weights, preprocessing and requested output contract. Use matched
source, threading/cache conditions, convergence requirements and repeated
unprofiled workers. Optimize smoothing in both arms or fix it in both arms;
do not compare an oracle-tuned interaction fit with an automatically tuned
additive denominator. A stopped fit does not supply a time to the required
solution. Tuning, restarts and screening belong in the full-pipeline total even
when each individual fit looks cheap.

RSS ratios need the retained-byte census beside them: a large common runtime
and input footprint can conceal substantial growth in model state. A difference
between process high-water marks is not by itself the number of interaction
bytes, because the peaks can occur in different phases. Full covariance or
all-term prediction exports can also dominate an otherwise cheap fit; use the
same requested outputs in both arms and state any deliberately restricted
uncertainty contract.

### Validation must span many datasets

The user's expanded requirement is a multi-dataset assessment, not a larger
number of repeats of this synthetic law. A separate corpus investigation owns
the dataset inventory and source validation. The budget comparison should span
datasets with mostly additive signal, a few local interactions, shared smooth
interaction structure and heterogeneous interactions, as well as differing row
counts, dimensions and predictor correlations. Real-data behavior is essential;
controlled synthetic cases remain useful for known mechanisms and refusals.

For each dataset, fit the additive and interaction arms on identical training
rows and parent inputs, with preprocessing learned from training data. Use the
same appropriate split and loss within that dataset; retain an untouched test
portion after representation, rank and smoothing choices. Keep training and
validation selection costs in the total. The current fixed-Gaussian reuse
hypothesis should first be assessed across datasets in that eligible regime;
changing-weight families require a separate validity and performance claim.

Compute normalized ratios within each dataset and mode before summarizing
across datasets. Report the distribution of those paired ratios, loss changes,
memory and the fraction of cases meeting each plotted budget/quality band.
Show adverse cases, convergence failures, unsupported/refused cases and cases
whose rich reference cannot finish; do not average only successful favorable
fits. A rich reference that fails to run cannot supply a verified
rich-model-fidelity target. Distinguish repeat-to-repeat timing variation from
variation across datasets, and report absolute additive times so tiny
denominators remain visible.

An improvement on the separable synthetic fixture is evidence for that regime.
It becomes a claim about useful cheap interactions only when the declared
statistical and resource tradeoff survives the broader corpus, including the
cost of selecting the representation.

The current evidence supports a bounded claim: dozens of small interactions
are already within a few additive-fit times with fixed smoothing and near the
exploratory scale with automatic smoothing on this fixture. Exact geometry
reuse is a concrete next attempt to improve that result. Much richer or more
numerous interactions require an explicit structural/statistical assumption,
whose accuracy and complete selection/fitting cost remain to be established.
