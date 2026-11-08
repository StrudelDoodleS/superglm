# SuperLSS family framework and distributional-regression roadmap

> **Archived planning snapshot.** P1 has since been implemented and locally validated. See the [implementation validation](../../benchmarks/declared_scalar_families_validation.md) for delivered scope and remaining acceptance items. Original proposal language and unchecked tasks below retain their historical planning status.

**Status:** research-backed proposal for owner review, 2026-11-08. No implementation is approved by this document. It complements, and does not replace, `notes/ROADMAP.md`.

**Scope:** SuperLSS/distributional regression only. Easier family development, large-loss severity (especially Lomax), and a bounded investigation of Dirichlet responses. Extending the scalar SuperGLM family catalogue, compositional input features, a general solver rewrite, and release work are excluded.

**Analysis pin:** remote `origin/master` at **`6643ed9c94497eba0efb8869b945a17c5d380caf`**, “Avoid repeated work in binned Gram assembly (#608)”. `git fetch origin master` confirmed the supplied pin on 2026-11-08; there was no intervening change to assess. All code assertions below come from `git show`/`git grep` at that SHA, not the older checkout HEAD (`c1493d7126e92b98a49458f019f8cfde04d03e32`). Repository links are pinned for that reason. Links between the two new documents are relative.

**Companion:** [First-stage implementation proposal](2026-11-08-superlss-family-framework-plan.md). It specifies modules, proposed interfaces, tests and acceptance gates. This note owns the research rationale, priorities and later choices.

## 1. Recommendation and decision status

**Recommendation:** finish the existing declarative severity infrastructure by supplying a small, production-quality family adapter. Use **Lomax first and log-logistic second** to demonstrate that adding a law does not require copying an entire family implementation. Retain the current symbolic generator, stable primitives, compiled row kernels, observation-window machinery and generic fitting contracts. Use SciPy as an interoperability and independent-reference layer, not as a source of regression derivatives.

Deliver useful severity modelling before opening a public arbitrary-distribution factory. Start with named, serializable family classes backed by an internal declarative specification. Once two families establish the boundary between reusable mechanics and family-specific mathematics, review a public declaration API. A registry of SciPy names alone would hide precisely the difficult parts: parametrisation, high derivatives, tails, observation mechanisms, weights and finite functionals.

**Recommended sequence:**

1. **P1 — implementation candidate:** adapter plus Lomax, with log-logistic as the second-law proof; exact and qualified censored/truncated fitting, honest derivative capabilities and moment behaviour.
2. **P2 — modelling/workflow candidate:** evaluate these against existing families, add tail/layer evaluation and threshold-sensitivity recipes, then promote Burr XII if it earns its extra parameter. Scope a fixed-threshold splice wrapper separately against issue #570.
3. **D1 — bounded feasibility spike:** investigate a vector-response Dirichlet extension after P1. It need not wait for every severity candidate. Beta LSS is an optional scalar-response control, not a prerequisite for all Dirichlet work.
4. **P3/D2 — conditional work:** choose Dagum, GB2, a public family factory, or application-specific Dirichlet delivery from evidence and the owner's use case. Do not implement the whole catalogue by default.

These are dependency and evidence stages, not calendar estimates. **The owner has selected the topic and priorities, not a dataset, an application, these interfaces, or an implementation.** All design choices in this note are recommendations. The planning task itself needs no further answers.

### Choices for the owner at the next implementation review

| Choice | Recommended starting position | What changes if the choice differs |
| --- | --- | --- |
| First severity use case | Claim-level positive losses or threshold excesses; identify which before claiming business validation | Ground-up losses, payments above deductibles, capped payments and excesses imply different likelihoods and functionals |
| First public deliverable | Named `LomaxLSS` and `LogLogisticLSS`, internal shared specification | An arbitrary user-authored factory adds validation, reproducibility, packaging and compatibility obligations |
| Lomax parametrisation | Positive scale and positive tail exponent, log links; unrestricted mathematical moment domain | Direct mean regression requires an explicit finite-mean restriction and defines a different regression model |
| Practical success criterion | Held-out tail calibration and expected bounded-layer losses, with complete-fit cost | A pricing-mean, reserving, capital or quantile objective can select different models and admissible moment domains |
| Dirichlet application | Choose continuous allocation shares versus loss-development/reserving versus category counts | Determines observation shape, missingness, zero policy, total-loss model and diagnostics; a generic Dirichlet GAM does not solve all three |
| Dirichlet interpretation | Alpha-coordinate feasibility first; mean/precision as a separate modelling decision | Mean/precision needs identifiable coupled transformations; it is not a cosmetic naming change |

## 2. What exists at the pinned master

| Verified state | Consequence for this proposal |
| --- | --- |
| Nine public families: `GaussianLS`, `GammaLS`, `LogNormalLS`, `NegativeBinomialLS`, `TweedieLSS`, `GeneralizedGammaLSS`, `GeneralizedParetoLSS`, `TwoPieceNormalLSS`, `TwoPieceLogNormalLSS`. [Family development guide][family-guide] | No scalar catalogue expansion and no claim that severity starts from zero |
| The solver-facing interface has `parameters`, `bind_likelihood`, `initialize`, `evaluate_natural`. Complete fitting additionally needs `to_config`. Parameter count is generic; a four-parameter scalar-response extension is tested. [Contract][family-contract], [extension tests][extension-tests] | Reuse the existing protocol; a four-parameter GB2 would not require a four-parameter solver redesign |
| Public `bind_predictor(family, name, *terms, ...)` supports custom parameter names. [Binding][binding] | Generic families need no generated convenience method for every possible name |
| `tools/generate_severity_families.py` is production machinery. It derives density/tail jets through order four where declared, emits Numba code with lazy orders, stable primitives and checked-in modules, and offers `--check`. [Generator][generator] | Extend it; do not reproduce the 2026-10-31 prototype as a second generator |
| Generated density modules are called by existing Gamma, log-normal, generalized Pareto and two-piece kernel routes; ordinary-row dispatch has range checks and wider array fallbacks. [Generated dispatch][declared-dispatch] | Generation is already on the fit path. Performance and numerical compatibility are real constraints, not hypothetical future work |
| Incomplete-gamma and generalized-gamma tail engines and shared declared-window composition exist. [Window numerics][declared-bounds], [kernel directory][kernels] | The older research note's unfinished-work list is partly stale; do not schedule these engines as new work |
| Weibull, mean-Weibull, Burr XII, log-logistic and Lomax are test declarations, not public built-ins. Strict xfails remain for fourth log-CDF derivatives of Burr XII and Lomax at `F=1e-100`. [Fixtures][severity-fixtures], [tests][severity-tests] | Tested expressions are starting evidence, not admission certificates for a new production family |
| A generic `DeclaredFamily` exists only in the research prototype; it uses a mutable registry, positive/log-linked parameters, complete observations and SciPy-based starts/predictions. [Prototype][prototype] | Extract lessons, not the implementation wholesale; production needs immutable identity, correct capabilities, bounds and save/load behaviour |
| GPD permits `0 <= shape_lower < shape_upper <= 1`, with fitted xi strictly interior. Negative xi and xi at/above one are excluded. Censoring/truncation and shape-wall disclosures exist. [GPD adapter][gpd] | Lomax overlaps an existing row law, but unrestricted positive tail exponents also reach infinite-mean regimes unavailable in this GPD adapter |
| Default prediction already preserves legitimate infinite means. Named posterior functionals reject non-finite summaries; a callable posterior quantity already works. [Model][model], [posterior][posterior] | Do not restrict a new law merely to keep `predict()` finite; finite layers can use existing callable propagation |
| `DistributionalRowState` and null fitting explicitly require a one-dimensional response. [Fit state][fit-state], [null model][null-model] | Dirichlet needs response plumbing work, independently of the already generic number of predictors |

The source search found no named `layer_price`, `layer_cost` or `threshold_stability` API in `src/superglm` at the pin. This is a bounded search result, not a claim that no custom recipe is possible. [Issue #570](https://github.com/StrudelDoodleS/superglm/issues/570) is an open proposal for a fixed-threshold object holding body, exceedance and tail fits.

The [existing strategic roadmap][existing-roadmap] still prioritizes previously selected interaction/scaling work. This request authorizes a new LSS proposal, not a silent replacement of that strategy. The [older severity note][older-severity] and [large-loss sweep][older-tail] remain valuable discovery material, but neither their status labels nor their prototype timings are current-head measurements. No fit or benchmark was run for this note.

### Reliability is a delivery condition

Open issues were read on 2026-11-08: [#604](https://github.com/StrudelDoodleS/superglm/issues/604) concerns stopping and penalty-geometry problems, including an LSS Gamma case; [#586](https://github.com/StrudelDoodleS/superglm/issues/586) documents slow smoothing paths and distinct plain-EFS solutions; [#593](https://github.com/StrudelDoodleS/superglm/issues/593) concerns real curvature jumps in two-piece laws. Their observations are tied to their own revisions and workloads, not reproduced here.

New smooth severity families should exercise the existing default stationarity finish, explicit EFS and several starts. Report coefficient convergence, smoothing status, rank, warnings, derivative authority and requested outputs separately. A family addition must not suppress a warning or relax an endpoint criterion to appear successful. Conversely, it need not absorb an unrelated solver unification: isolate a reproducible solver blocker and route it to that workstream. The repository's current policy is to disclose imperfect convergence, not refuse a fit solely for lacking certification. [Pinned working agreements][remote-agents]

## 3. Literature implications for a family factory

Distributional regression models several parameters of the conditional response law, each potentially with its own predictor. The GAMLSS lineage is the appropriate starting point, rather than scalar exponential-family membership. The original reference is [Rigby and Stasinopoulos (2005)](https://doi.org/10.1111/j.1467-9876.2005.00510.x); its bibliographic record and the [authors' package documentation](https://search.r-project.org/CRAN/refmans/gamlss/html/gamlss-package.html) were verified, but the original full text was inaccessible in this review.

[Wood, Pya and Säfken (2016), sections 2–3.2](https://arxiv.org/abs/1511.03864) give a general penalized-likelihood/Laplace framework with four continuous likelihood derivatives and explain parameter-to-predictor derivative conversion. [Wood and Fasiolo (2017), general-likelihood development](https://arxiv.org/abs/1606.04802) offer generalized Fellner–Schall updates using first and second derivatives. Their non-Gaussian simplification neglects a Hessian-dependence term; it does not make every EFS endpoint an exact LAML optimum. **Design inference:** generate higher derivatives when affordable, but do not make all fitting depend on their availability or confuse the routes' guarantees.

### Required work depends on the fitting route

| Route or feature | Mathematical information needed | Existing SuperLSS implication |
| --- | --- | --- |
| Fixed smoothing parameters, observed Newton coefficients | Log likelihood, score, raw signed Hessian; appropriate local differentiability | `evaluate_natural(..., derivative_order=0/1/2)` is the core contract; cheap order-zero trials must represent the same likelihood |
| Fisher scoring/fallback | Expected information for the actual observation law | Optional protocol, not a replacement for the raw Hessian. Current provider selection excludes censored/truncated plans |
| Plain generalized EFS update | First/second derivatives for its update | Endpoint diagnostics or other requested features can still invoke higher-order information; “EFS” is not a guarantee that the whole call uses order two only |
| Matched LAML gradient and stationarity calculations | Third likelihood derivatives after link composition | Exact directional protocol exists; otherwise the engine differences curvature with evidence |
| Newton LAML Hessian and smoothing-uncertainty correction | Fourth derivatives and inverse-link derivatives through order four | Exact second-direction protocol exists; otherwise certified finite-difference machinery is available, with weaker/conditional authority |
| Censored/truncated versions of these routes | Corresponding derivatives of log observation/truncation probabilities | A density-only generator does not supply these automatically |
| Expected shortfall | A finite tail moment and a justified evaluation method | `ExpectedShortfallFamily` is independent of CDF/quantile capability; endpoint-singular quantile quadrature is not certification |

The numerical and protocol details above come from the [family contract][family-contract], [development guide][family-guide] and [smoothing derivatives][smoothing-derivatives]. `FamilyCapabilities.max_derivative_order` currently describes the core natural evaluation (typically two); higher directional methods are separate protocols. Do not set it to four while `evaluate_natural` accepts only zero through two. Similarly, a four-times differentiable symbolic expression is not a guarantee of stable fourth derivatives in float64.

### Three credible approaches

| Approach | Strength | Cost or limitation | Recommendation |
| --- | --- | --- | --- |
| Extend the existing SymPy/stable-primitive generator plus a shared adapter | Reuses shipping kernels, packed derivatives and compiled CPU execution; explicit generated provenance; no symbolic runtime dependency | Special functions, limits, tail cancellation and support-dependent branches still require mathematics and tests | **Primary production route** for the first severity families |
| SciPy interoperability plus explicit derivative providers | Familiar distribution names, reference densities/CDFs/quantiles/simulation, optional initial values; useful bridges to external tools | Distribution APIs do not provide SuperLSS regression scores, Hessians, higher derivatives, weighting or observation contracts | **Complement**, initially explicit per-family mapping with `loc=0` |
| Numerical differentiation or an AD backend | Quick research prototypes; an independent derivative oracle; potentially attractive for complicated smooth likelihoods | Cost, scaling, special-function derivatives, branch smoothness, float64 range, platform/JIT dependencies and certified authority need separate evidence | **Research fallback**, not a new mandatory runtime stack |

[SciPy 1.15.2 `make_distribution`](https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/scipy.stats.make_distribution.html) adapts an `rv_continuous` instance into the newer distribution interface. The [current documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.make_distribution.html), labelled 1.18.0 when viewed, also describes compatible custom objects and a versioned protocol. Those custom-object capabilities cannot be assumed at SuperGLM's `scipy>=1.15` floor. Neither interface promises regression derivatives. Its scope is univariate, so it is not the route to a Dirichlet response.

[SciPy `rv_continuous.fit`](https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/scipy.stats.rv_continuous.fit.html) can estimate constant distribution parameters and accepts censored data. That is useful as a cross-check, not as a penalized distributional-regression implementation; it also does not establish SuperLSS's truncation normalization or frequency-weight identities. Always pin location where the intended severity law fixes it. A failed SciPy start must not become an unexplained family refusal.

AD has credible precedent: [Kristensen et al. (2016), TMB](https://arxiv.org/abs/1509.00660) use AD through order three in Laplace-based modelling. That establishes feasibility of the technique, not compatibility with this engine's fourth-order, tail and certification contracts. Our engineering inference is to compare a small optional AD prototype only if generated expressions become unmanageable, especially for later coupled transforms. Differentiating an unstable expression preserves its instability; finite differences additionally trade truncation against cancellation and must not cross support boundaries. Complex-step differentiation is not a universal alternative when a likelihood or library primitive is non-holomorphic or rejects complex inputs.

**Dependency policy:** NumPy/SciPy/Numba stay runtime dependencies; SymPy and mpmath remain development/generation/oracle dependencies already present in the dev setup. Generated source ships in the wheel. No import of `tools`, research code, SymPy or mpmath from a runtime family. No SciPy floor increase or AD dependency is part of P1. [Pinned dependency declarations][pyproject]

### What the declaration should actually describe

Keep separate, but checked for consistency:

1. **Law and coordinates:** ordered names, domains, fixed response support, density/tail expressions, coordinate convention and stable primitives. This is mostly the existing build-time `Declaration`.
2. **Regression adapter:** parameter metadata/default links, initialization, immutable likelihood-plan identity, supported weights, observation admission and configuration/serialization.
3. **Optional operations:** CDF/quantile, mean, variance, random sampling, Fisher information, window probabilities, higher curvature directions and expected shortfall. Each needs a real implementation and domain policy. An absent protocol must remain absent; a generic stub that always raises can accidentally advertise a structural capability.
4. **Evidence:** law and parameter mapping, derivative oracle, domain/roundoff envelope, unsupported cases, regeneration check, fit comparisons and actual dispatch.

The factory should remove mechanical repetition, not manufacture unsupported mathematics. Family-specific callbacks for stable tails, initialization or moment evaluation are a desirable escape hatch. No mass migration of the nine existing families is needed to validate this design.

## 4. Severity choices: payoff, parametrisation and moments

Use tail exponent `t` to mean `S(y) ~ C y^(-t)`, not the EVT index `xi=1/t`. All scales below are positive and location is fixed at zero. Moment conditions here are for positive moments `r>0`; equalities are divergent boundaries. The conditions and nesting relations are mathematical deductions from the linked densities, rather than benchmark findings.

| Family | Useful parametrisation and survival/CDF | Positive moment condition | Payoff and readiness |
| --- | --- | --- | --- |
| **Lomax** | Scale `lambda`, tail exponent `alpha`; `S(y)=(1+y/lambda)^(-alpha)` | `r < alpha` | Explicit user priority; simple, interpretable tail; existing test declaration. **First implementation candidate**. [SciPy law](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lomax.html) |
| **Log-logistic/Fisk** | Scale/median `s`, exponent `a`; `S(y)=1/(1+(y/s)^a)` | `r < a` | Two-parameter power-tail alternative with a body shape distinct from Lomax; existing test declaration. **Second pilot**. [SciPy law](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisk.html) |
| **Burr XII** | Scale `s`, body shape `c`, second shape `k`; `S(y)=(1+(y/s)^c)^(-k)` | `r < c*k` | Separates body flexibility from tail decay; contains Lomax at `c=1` and log-logistic at `k=1`. Test declaration exists. **Next implementation candidate after identifiability/evaluation gate**. [SciPy law](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.burr12.html) |
| **Dagum/Burr III** | Scale `b`, exponent `a`, body parameter `p`; `F(y)=(1+(y/b)^(-a))^(-p)` | `r < a` | Alternative left-body behaviour; tail exponent is `a`, not `a*p`. No inspected production declaration. **Research/comparison candidate**. [SciPy law](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.burr.html) |
| **GB2** | Scale `b`, power `a`, beta shapes `p,q`; transformed `Z=(Y/b)^a/(1+(Y/b)^a) ~ Beta(p,q)` | `r < a*q`; `E[Y^r]=b^r B(p+r/a,q-r/a)/B(p,q)` | Broad four-parameter umbrella: Burr XII at `p=1`, Dagum at `q=1`. Density differentiation is easier than stable incomplete-beta parameter tail derivatives. **Later research gate**. [GAMLSS documentation](https://gamlss-dev.r-universe.dev/gamlss.dist/doc/manual.html#GB2) |
| **Weibull** | Scale `s`, shape `k`, or mean `m=s Gamma(1+1/k)` | All positive moments | Reduced-family/control value, rather than new support or power-tail coverage; nested in existing generalized gamma. Test declarations already exist. **Control before public promotion**. [Existing generalized gamma][gg] |

GB2 has real insurance precedent: [Frees and Valdez (2008)](https://doi.org/10.1198/016214508000000823) use it for the severity component of hierarchical automobile claims modelling. Only the publisher's abstract was read here. This supports considering GB2; it does not establish that four covariate-varying parameters beat simpler severity models on the owner's book.

### Why a dedicated Lomax still matters

For a Lomax law with `(lambda, alpha)`, the equivalent GPD parameters are

\[
\xi=1/\alpha,\qquad \sigma=\lambda/\alpha.
\]

Substitution in the [GPD density](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genpareto.html) verifies row-law equivalence. The currently supported positive GPD range corresponds to `alpha>1` (or a smaller interval under configured walls). Allowing all `alpha>0` in Lomax adds **moment-domain coverage relative to this adapter**, not a new abstract distribution class.

A row-law identity does not imply the same fitted GAM. Existing GPD uses a bounded-logit shape link, while the proposed Lomax uses `log(alpha)`; under the default GPD walls, `logit(xi)=-log(alpha-1)`, which is nonlinear in `log(alpha)`. Even where two log-coordinate changes are linear, mixing predictors changes term restrictions and usually the separable penalties. Compare likelihoods and predictions after parameter transformation, not equality of spline coefficients or smoothing parameters. This is a deduction about the specified regression models.

Recommend `(scale, shape)=(lambda, alpha)` for the first public Lomax, with roles documented as monetary scale and tail exponent. Offer no fitted location: a free location would make response support parameter-dependent. A later mean form `lambda=mu*(alpha-1)` is useful for multiplicative pricing effects but necessarily restricts `alpha>1`; it must be an explicit parametrisation, not hidden clipping.

Lomax includes `y=0` with finite density. The fixture's log-space expression needs a checked zero-limit path before public admission. Rounded zeros are not evidence for a structural atom. Log-logistic/Burr can have singular or zero density at zero depending on shape; start their exact-response contract on `y>0`, while allowing observation windows to touch zero. Document the observation interpretation instead of inserting an epsilon.

### Parameter domains follow the requested quantity

For Lomax:

\[
E[Y]=\lambda/(\alpha-1)\quad(\alpha>1),\qquad
\operatorname{Var}(Y)=\lambda^2\alpha/[(\alpha-1)^2(\alpha-2)]\quad(\alpha>2).
\]

For `0<p<1`, `q_p=lambda*expm1(-log1p(-p)/alpha)` is mathematically finite for every `alpha>0`. The upper conditional tail mean is

\[
ES_p=q_p+\frac{\lambda+q_p}{\alpha-1},\qquad \alpha>1,
\]

and is infinite otherwise. These follow directly by integrating the survival function. Distinguish a divergent population moment from a finite value overflowing float64. The latter needs a range explanation, not a claim that the model has infinite mean.

For a bounded layer with attachment `a>=0` and finite width `l>=0`, let `A=1+a/lambda` and `B=1+(a+l)/lambda`. Then

\[
L(a,l)=E[\min\{(Y-a)_+,l\}]=\int_a^{a+l}S(y)\,dy
=\begin{cases}
\lambda\{B^{1-\alpha}-A^{1-\alpha}\}/(1-\alpha),&\alpha\ne1,\\
\lambda\log(B/A),&\alpha=1.
\end{cases}
\]

**Every such layer is finite and lies in `[0,l]`, even when `alpha<=1`.** Implementing the displayed subtraction literally near one would be unstable: use `log1p`/`expm1` or an equivalent scaled limit, and test continuity at one and narrow layers. Unbounded stop-loss and expected shortfall still need a finite mean. A severity layer is an expectation per claim, not an aggregate portfolio premium; occurrence probability/frequency, dependence, inflation and financial loadings remain separate.

For future Burr XII, consider modelling `(scale, body_shape, tail_exponent)` with `k=tail_exponent/body_shape`. This makes the moment condition transparent. It is a proposed regression parametrisation requiring its own chain rules and validation, not a relabelling of the SciPy parameter `d`. Likewise GB2 should not receive arbitrary independent links and an implicit `a*q>1` check that changes the fit unnoticed. Explicit unrestricted and moment-constrained parametrisations are alternative models.

### Identifiability and model size

Start comparisons with constant tail shape and a modest scale predictor, then add a small tail predictor only if it improves held-out quantities. Smooth scale and shape may compensate for each other; heavy censoring and few exceedances make the problem worse. Burr/GB2's extra body and tail parameters increase this risk. Assess likelihood profiles, observed/Fisher geometry where available, multi-start results, uncertainty and sensitivity to basis size and threshold. A successful optimizer flag is not evidence that a rich tail predictor is identified.

Weibull and existing generalized gamma/log-normal are meaningful controls, including when a power tail is unnecessary at the observed layers. A reduced family can improve variance and interpretation even when it adds no new row laws. Avoid choosing a family solely because it is nested in a larger one or has a good bulk score.

## 5. Observation laws, tail workflows and evaluation

### Censoring and truncation are first-class contracts

For an observed exact value in truncation region `T_i`, a frequency-weighted contribution is `w_i*(log f(y_i)-log P(T_i))`. For a censored observation region `O_i`, it is `w_i*(log P(O_i intersect T_i)-log P(T_i))`. A cap is usually right censoring of the ground-up loss; absence below a reporting threshold is truncation. Treating capped payments as exact ground-up losses biases the likelihood.

The current public `censored=` contract admits left- and right-censored rows, alongside exact rows and per-row truncation. It does not admit a general interval-censored observation with two independent observation endpoints. P1 retains that contract. Arbitrary finite intervals are still needed inside probability/window numerics; supporting them numerically does not add a public interval-censoring observation law. General interval censoring requires separate observation-API scoping. [Pinned observation contract](https://github.com/StrudelDoodleS/superglm/blob/6643ed9c94497eba0efb8869b945a17c5d380caf/src/superglm/distributional/observation.py)

Use existing `ObservationContract`, `ResponseBounds`, immutable row selection, `bounded_channels`, `bounded_higher_derivatives` and declared window composition. Compute windows from log tails and stable differences; use the existing narrow-window integration mechanism only with a justified family-specific rate bound. Density derivatives alone are insufficient. [Observation adapters][response-bounds], [window numerics][declared-bounds]

Weights must retain their existing meaning. Integer frequency weights replicate complete row contributions, including carriers and window normalizers, at every derivative order. They do not sharpen the individual loss distribution. Non-unit prior weights change the row law for reproductive families; the first severity pilots should explicitly reject them because no such law has been supplied. Do not silently treat exposure as replication. The generic weight resolver owns dropped zero-weight rows and provenance. [Weight contract][weights]

The strict fourth-log-CDF xfails are a specific warning: a finite zero can still be an incorrect derivative. A production higher-order route must stabilize those states, route them to a validated alternative, or decline the analytic capability there so existing fallback/evidence applies. Removing an xfail without a regression/mutation demonstration is not progress. [Current tests][severity-tests], [issue #564](https://github.com/StrudelDoodleS/superglm/issues/564)

### Full severity and peaks over threshold answer different questions

Lomax/Burr/GB2 can model an entire conditional severity distribution; GPD excess modelling is an asymptotic tail strategy with a threshold decision. [Youngman (2022)](https://doi.org/10.18637/jss.v103.i03) gives a directly relevant GAM extreme-value implementation. [Northrop, Attalides and Jonathan (2017)](https://arxiv.org/abs/1504.06653) compare training thresholds at a common higher validation threshold and study threshold uncertainty. Their application is ocean storms, not an insurance validation result.

Recommended P2 evidence: refit across a training-defined threshold grid, track tail parameters and target layers, retain exceedance counts, and compare predictions for the same held-out target event. Under a genuine GPD law above `u`, the threshold transform is `sigma_v=sigma_u+xi*(v-u)` with unchanged xi. This is a diagnostic identity, not a guarantee that fitted regression curves or their uncertainty remain invariant under refitting.

[Reynkens et al. (2017)](https://arxiv.org/abs/1608.01566) study body-tail splicing with censoring and truncation. Their EM treatment is especially relevant when an observed interval crosses the splice: the component is then unobserved. Consequently, issue #570's separate-fit factorization should be scoped to its fixed-threshold, known-component recipe; it must not be generalized to arbitrary cross-threshold censoring. A splice can have a continuous CDF without matching densities at the join. Neither a fixed-threshold wrapper nor a family factory automatically chooses thresholds or estimates threshold uncertainty.

### Evaluation gate beyond a bulk log score

Predeclare data splits, thresholds, attachments, limits and predictor complexity using training/validation data. Keep a final test set untouched; use time or group splits when policies, events or development years are dependent. No application dataset has yet been selected. freMTPL2 is useful for continuity with existing LSS evidence, but is not automatically a well-powered large-loss/censoring benchmark.

Report:

- **Whole observed-law log score**, accounting for its censoring/truncation mechanism, and the complete-fit diagnostics alongside it.
- **Tail probability calibration** at common attachments/thresholds, exceedance counts and uncertainty; tail quantile coverage and pinball loss on observable outcomes.
- **A proper tail-focused score.** For complete observations and a fixed threshold `v`, the negative censored log score is `-1(y>v)*log f(y)-1(y<=v)*log F(v)`. It assesses both tail mass and tail density. A conditional-exceedance score alone cannot check exceedance probability. Do not simply multiply log scores by realized loss or retain only extreme outcomes. [Diks, Panchenko and van Dijk (2011)](https://doi.org/10.1016/j.jeconom.2011.04.001), [Allen (2023), weighted scoring rules](https://arxiv.org/abs/2305.07312)
- **Layer outcomes:** predicted `L(a,l)` versus observed bounded payouts, calibration by exposure/risk group and attachment, aggregate actual/expected with uncertainty when the denominator is meaningful, and squared error for bounded layer means. Use only layers identifiable from the recorded payment/observation data. A cap below the layer does not reveal its payout; simulated latent losses are a simulation oracle, not observed validation.
- **Decision stability:** changes in layer means, quantiles and finite expected shortfalls across families, thresholds, smoothing starts and sensible tail complexity. Separate parameter uncertainty, threshold/model uncertainty and future-loss variability.

CRPS or other moment-dependent summaries also need their existence conditions checked. For infinite-mean candidates prefer log scores, exceedance probabilities and bounded functionals unless a particular scoring integral is proved finite. Existing scalar callable `posterior_bounds` can propagate a finite layer functional; it does not automatically account for model selection, threshold selection or a future aggregate loss process. Never drop posterior draws crossing a moment boundary and report the remaining expected-shortfall average as if nothing happened.

## 6. Dirichlet: an explicit feasibility project

### Applications and the limit of the evidence

The strongest directly relevant actuarial source is [Sriram and Shi (2021)](https://doi.org/10.1111/jori.12311), with [author preprint](https://arxiv.org/abs/1904.04609). Its model treats development allocations as Dirichlet and relates reserving predictions to Chain–Ladder/Bornhuetter–Ferguson depending on available information. Sections 2–3 of the full preprint were inspected: it uses positive incremental payments, unknown accident-year scale and a remaining-tail component; incomplete development and external tail information matter. **A SuperLSS Dirichlet response over known shares is a building block, not that entire reserving model.** Unknown ultimate totals induce additional estimation/support issues; normalizing a partial triangle by its observed total does not recover missing development shares.

Other plausible uses are continuous positive shares of a known total loss across coverage components or settlement categories. These are proposed applications, not validated SuperLSS examples. They require a separate total-loss model if absolute money is needed. Conditional raw-share covariance is restricted by the Dirichlet law, so it cannot be presented as general multivariate severity dependence.

[Gan and Valdez (2024)](https://variancejournal.org/article/116404-compositional-data-regression-in-insurance-with-exponential-family-pca) study compositional **predictors** and EPCA in an insurance-related injury-count application. Their full-text introduction/method setup confirms that distinction. It motivates care with compositions but is not evidence for a Dirichlet response family. Compositional feature processing is outside this roadmap.

### Two modelling choices, not one relabelled interface

For a response `y_i` of length `K` with positive entries summing to one, write `alpha_0=sum_j alpha_j`:

\[
\ell_i=\log\Gamma(\alpha_0)-\sum_j\log\Gamma(\alpha_j)
       +\sum_j(\alpha_j-1)\log y_{ij}.
\]

Differentiation gives `s_j=psi(alpha_0)-psi(alpha_j)+log y_j` and `H_jk=psi1(alpha_0)-1(j=k)*psi1(alpha_j)`. Higher natural derivatives follow by further polygamma differentiation. This deduction supplies a plausible analytic kernel, not evidence that vector fitting already works. [SciPy Dirichlet definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dirichlet.html)

1. **Independent alpha predictors:** `alpha_j=exp(eta_j)`, one predictor per component. Scalar link composition can be reused. Means `mu_j=alpha_j/alpha_0` and precision `phi=alpha_0` are derived; each predictor generally changes both. This is the smallest engine-feasibility target.
2. **Mean-composition/precision:** `alpha_j=phi*mu_j`, `phi=exp(zeta)`, with `K-1` identified mean logits and a reference component (or an explicit contrast). There are still `K` free predictor coordinates, not `K+1` independent ones. The map is coupled. One option is a family-owned likelihood in `(K-1 logits, precision)` using identity links for logits and a log link for precision, with all cross derivatives supplied; another is a general coupled-transform protocol. Prefer investigating the family-owned option before expanding the engine's link system.

Both parametrisations are established in [Maier (2014)](https://research.wu.ac.at/en/publications/dirichletreg-dirichlet-regression-for-compositional-data-in-r-3/) and [DirichletReg's official package reference](https://mirror.hi.no/cran/web/packages/DirichletReg/refman/DirichletReg.html). Different predictor restrictions and penalties produce different models. Reference-component invariance of an unrestricted law does not guarantee invariance under independently chosen penalized smooths.

**Beta LSS** with mean/precision is a useful scalar-response stepping stone for a single continuous fraction; [Ferrari and Cribari-Neto (2004)](https://doi.org/10.1080/0266476042000214501) provide the foundational regression parametrisation. It exercises unit-interval support and boundary policy, and Dirichlet at `K=2` provides a reduction oracle. It does not test the vector-response plumbing or prove a multivariate diagnostic design. Censored beta/GB2 tails would require an incomplete-beta derivative capability, distinct from the existing incomplete-gamma engine.

### Response plumbing to audit and reuse

Audit `fit_state.py`, `null_model.py`, family response validation, solver entry validation, observation/weight row counts, public `fit`/prediction methods, retained rows, cross-validation, serialization, scoring, residuals and plots. Preserve row count `n` independently of response dimension `K`; weights are per observation, not per component. Likelihood output remains one scalar per row, score `(n,K)`, packed Hessian `(n,K*(K+1)/2)`. Existing design stacking and coefficient assembly are candidates for reuse. Whether every path is response-neutral is an explicit spike question; two one-dimensional guards are evidence of a boundary, not an exhaustive change list.

**Proposed D1 boundary:** complete strictly positive compositions, fixed component names/order, `K=3` synthetic data and `K=2` beta reduction; independent log-alpha predictors; unit prior or integer frequency weights; no censored components, unknown totals, structural zeros or production API guarantee. No demand for a new multivariate CDF, scalar quantile, scalar PIT or fake univariate residual. Return a response-shape impact map, a small feasibility fit against an independent likelihood, and a recommendation to proceed or stop.

Acceptance would require correct simplex validation with a dimension/float64-based sum tolerance (no silent large renormalization), preserved column identity, frequency replication, finite natural/predictor derivatives, stable fitted mean compositions, component permutation equivariance under equivalently permuted designs/penalties, null-fit and save/load round trips, and disclosure on nearly degenerate compositions. Test near-boundary diagnostics using stable outputs; do not demand coefficient recovery where concentration is poorly identified.

For diagnostics, investigate joint log score, component/aggregated-share calibration and simulation-based composition discrepancies. Beta marginal PITs are legitimate marginal checks but are dependent and not a single joint PIT. Sequential conditional transforms require an explicit component order. Conditional Dirichlet raw shares have `Cov(Y_j,Y_k)=-mu_j*mu_k/(phi+1)` for `j!=k`; positive conditional raw-share associations or richer log-ratio dependence motivate a logistic-normal or another model. [Aitchison and Shen (1980)](https://doi.org/10.1093/biomet/67.2.261) is a later alternative to examine; only its abstract was read.

### Zeros, counts and totals are application gates

- **Structural zeros:** impossible categories require a support/mixture or subcomposition model; a continuous Dirichlet with all positive parameters cannot encode their point mass.
- **Rounded/detection-limit zeros:** need a documented measurement model or explicitly justified transformation, with sensitivity analysis. Do not add a default pseudocount.
- **Sampling zeros in category counts:** a [Dirichlet-multinomial](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dirichlet_multinomial.html) is a different observation law on integer counts with known trial totals. Monetary shares are not counts merely because they can be written in cents.
- **Reserving zeros/negative increments/incomplete triangles:** classify their meaning before modelling. The inspected reserving paper assumes positive incremental payments; generic composition support does not address recoveries, negative development or unknown ultimate amounts.

## 7. Roadmap and evidence gates

| Stage | Deliverable and classification | Depends on | Exit evidence / reason to stop |
| --- | --- | --- | --- |
| P0 | **Research baseline**, this note and companion plan | Pinned source and primary-source review | Current/prototype states separated; owner choices exposed. Completed as documentation only |
| P1a | **Implementation candidate:** internal specification/adapter with complete-row Lomax | Owner approval of the first-stage proposal | Correct law/weights/carriers, lazy orders, immutable plan identity, fitting, predictions and serialization; no new solver family branches |
| P1b | **Implementation candidate:** stable window integration and directional capabilities; log-logistic second pilot | P1a and per-law numerical admission | Mixed observation tests; extreme fourth-derivative error either fixed or honestly routed/declined; second law uses the same mechanics |
| P1c | **Validation candidate:** full fit, reference and performance receipt | P1a/b | Independent intercept/penalized comparisons; several starts; numerical outputs, timing, peak RSS and actual backend. Scope down if targets cannot be represented honestly |
| P2 | **Modelling candidate:** tail/layer validation, threshold recipes; Burr XII and fixed-threshold splice scoped separately | P1 and a selected data/observation contract | Fresh held-out target evidence, tail sample size and uncertainty, identifiability, existing GPD/generalized-gamma/log-normal comparisons. No guaranteed model win |
| D1 | **Feasibility spike:** Dirichlet vector response | P1 contract lessons; an explicitly bounded spike request | Reusable-engine proof, response-shape impact map, beta reduction, boundary/zero policy and application recommendation. Can stop with a useful report |
| P3 | **Research then implementation candidates:** Dagum/GB2/public declaration API | Evidence that simpler families or internal extension workflow are inadequate | GB2 tail derivative and identifiability gate; factory capability/versioning/serialization design; no catalogue-only justification |
| D2 | **Conditional implementation:** selected Dirichlet parametrisation/application | D1 and owner selection of shares/counts/reserving | Suitable data, diagnostics, observation law, totals/zero handling; reserving needs its own specification |

P1's detailed tasks are in the [companion plan](2026-11-08-superlss-family-framework-plan.md). Full fits and native-platform checks are requirements for future numerical work, not work requested from the parent for this documentation handoff. No old timing is used as a current performance promise.

## 8. Unresolved questions and limits

1. **Engineering gate:** how much of the prototype adapter survives production plan identity, chunk reuse and serialization? Its mutable registry and implicit capabilities are not acceptable defaults. The proposed interfaces need an implementation review.
2. **Numerical gate:** stabilize or bound the affected extreme log-CDF channels before advertising analytic higher directions over them. Natural- versus log-coordinate evaluation may change the best route; a finite result alone is insufficient evidence.
3. **Modelling gate:** no dataset establishes whether Lomax, Burr, Dagum or GB2 improves a target layer. Parametrisation, tail flexibility and threshold must be evaluated together without test leakage.
4. **Observation gate:** arbitrary cross-threshold censoring does not inherit the simple splice factorization. Positive-xi unbounded support is much easier than negative-xi GPD support depending on fitted parameters; the latter remains separate research.
5. **Dirichlet gate:** the finite-dimensional score/Hessian is straightforward; vector validation, application meaning, zeros, totals and multivariate diagnostics are the risk. Exact higher-order cancellation at large concentration also needs measurement.
6. **Literature limit:** full texts were not obtained for every foundational source. Reading status is recorded below; inaccessible originals are not treated as reviewed derivations. No historical prototype was rerun and no current benchmark is claimed.

## 9. Annotated sources and reading record

`F` means the original full text was accessible and the specified sections were read; it does **not** mean every appendix was reviewed. `A` means abstract only. `D` means official documentation. `R` means source/issue inspection. Algebra explicitly marked as a deduction above is this proposal's reasoning. Repository prototype measurements remain historical evidence, not fresh experiments.

| Source | Evidence inspected | Use and limitation |
| --- | --- | --- |
| [Rigby & Stasinopoulos (2005), JRSS C 54:507–554](https://doi.org/10.1111/j.1467-9876.2005.00510.x); [GAMLSS reference](https://search.r-project.org/CRAN/refmans/gamlss/html/gamlss-package.html) | Bibliographic record + D; original full text unavailable | Framework lineage and distributional predictors; no unverified claim about the original algorithm's implementation details |
| [Wood, Pya & Säfken (2016), JASA, DOI 10.1080/01621459.2016.1180986](https://arxiv.org/abs/1511.03864) | F: sections 2, 3.1–3.2 and derivative discussion | Smoothness, LAML, parameter/link derivative requirements; not proof of SuperLSS implementation correctness |
| [Wood & Fasiolo (2017), Biometrics, DOI 10.1111/biom.12666](https://arxiv.org/abs/1606.04802) | F: introduction, update comparison and general-likelihood extension | Explains second-order EFS and its approximation; not a universal exact-stationarity guarantee |
| [Kristensen et al. (2016), TMB, JSS 70(5)](https://doi.org/10.18637/jss.v070.i05); [preprint](https://arxiv.org/abs/1509.00660) | F: abstract and methodological introduction | AD/Laplace precedent; published performance is not a SuperLSS benchmark |
| [Youngman (2022), evgam, JSS 103(3)](https://www.jstatsoft.org/article/view/v103i03/4317) | F: introduction and model overview | GAM extreme-value comparison route; environmental examples do not validate an insurance book |
| [Reynkens et al. (2017), IME 77:65–77](https://arxiv.org/abs/1608.01566) | F: introduction and censoring/splicing setup | Body-tail rationale and cross-threshold interval complication; not a ready-made regression wrapper |
| [Northrop, Attalides & Jonathan (2017), DOI 10.1111/rssc.12159](https://arxiv.org/abs/1504.06653) | F: introduction and predictive validation-threshold development | Common-target threshold comparison and uncertainty; Bayesian ocean-storm application |
| [Diks, Panchenko & van Dijk (2011), DOI 10.1016/j.jeconom.2011.04.001](https://repub.eur.nl/pub/25637) | A of published article; [2008 author working paper](https://papers.tinbergen.nl/08050.pdf) abstract/introduction; formulas cross-checked in Allen | Proper tail likelihood scores, caution about naive outcome weighting |
| [Allen (2023), Weighted scoringRules](https://arxiv.org/abs/2305.07312) | F: introduction and conditional/censored score definitions | Operational definitions for proposed evaluation; paper is not an actuarial validation |
| [Frees & Valdez (2008), JASA 103:1457–1469](https://doi.org/10.1198/016214508000000823) | A from publisher indexing; full-text access failed | Concrete GB2 insurance application; no inferred claim that all four parameters were modelled with smooths |
| [Sriram & Shi (2021), JRI 88:195–230](https://doi.org/10.1111/jori.12311); [2019 preprint](https://arxiv.org/abs/1904.04609) | A of published article; F: preprint introduction, sections 2–3, bootstrap caveat | Actuarial Dirichlet response motivation, incomplete development and latent-total qualification; version differences not exhaustively reconciled |
| [Maier (2014), DirichletReg, Report 125](https://research.wu.ac.at/en/publications/dirichletreg-dirichlet-regression-for-compositional-data-in-r-3/); [package reference](https://mirror.hi.no/cran/web/packages/DirichletReg/refman/DirichletReg.html) | A + D; report PDF inaccessible | Alpha and mean/precision parametrisations; potential external regression oracle |
| [Ferrari & Cribari-Neto (2004), J. Applied Statistics 31:799–815](https://doi.org/10.1080/0266476042000214501) | A from publisher and author-PDF indexing; full-text fetch failed | Scalar beta stepping stone; does not resolve vector response or boundary masses |
| [Gan & Valdez (2024), Variance 17(1), DOI 10.66573/001c.116404](https://variancejournal.org/article/116404-compositional-data-regression-in-insurance-with-exponential-family-pca) | F: introduction and method setup | Compositional predictors, specifically distinguished from response-family work |
| [Aitchison & Shen (1980), Biometrika 67:261–272](https://doi.org/10.1093/biomet/67.2.261) | A | Logistic-normal alternative; further comparison is a later research gate |
| [SciPy distribution reference](https://docs.scipy.org/doc/scipy/reference/stats.html), individually linked laws above; both `make_distribution` versions | D: signatures, parameter/support definitions and versioned interfaces | Parameter mapping and independent body/reference values; no guarantee of extreme derivative accuracy |
| [GAMLSS GB2](https://gamlss-dev.r-universe.dev/gamlss.dist/doc/manual.html#GB2); [SymPy code generation](https://docs.sympy.org/latest/modules/codegen.html) | D | Family nesting/reference parametrisation and generator tooling; no automatic numerical-stability guarantee |
| Pinned repository files below and issues #564/#570/#586/#593/#604 | R: code, tests, documentation, current issue bodies | Implementation inventory and known constraints; tests were inspected, not run |

### Pinned repository references

[remote-agents]: https://github.com/StrudelDoodleS/superglm/blob/6643ed9c94497eba0efb8869b945a17c5d380caf/AGENTS.md
[existing-roadmap]: https://github.com/StrudelDoodleS/superglm/blob/6643ed9c94497eba0efb8869b945a17c5d380caf/notes/ROADMAP.md
[family-guide]: https://github.com/StrudelDoodleS/superglm/blob/6643ed9c94497eba0efb8869b945a17c5d380caf/docs/development/internals/distributional-family-development.md
[family-contract]: https://github.com/StrudelDoodleS/superglm/blob/6643ed9c94497eba0efb8869b945a17c5d380caf/src/superglm/distributional/family.py
[extension-tests]: https://github.com/StrudelDoodleS/superglm/blob/6643ed9c94497eba0efb8869b945a17c5d380caf/tests/test_distributional_extensibility.py
[binding]: https://github.com/StrudelDoodleS/superglm/blob/6643ed9c94497eba0efb8869b945a17c5d380caf/src/superglm/distributional/binding.py
[generator]: https://github.com/StrudelDoodleS/superglm/blob/6643ed9c94497eba0efb8869b945a17c5d380caf/tools/generate_severity_families.py
[declared-dispatch]: https://github.com/StrudelDoodleS/superglm/blob/6643ed9c94497eba0efb8869b945a17c5d380caf/src/superglm/distributional/kernels/_declared/__init__.py
[declared-bounds]: https://github.com/StrudelDoodleS/superglm/blob/6643ed9c94497eba0efb8869b945a17c5d380caf/src/superglm/distributional/kernels/_declared_bounds.py
[kernels]: https://github.com/StrudelDoodleS/superglm/tree/6643ed9c94497eba0efb8869b945a17c5d380caf/src/superglm/distributional/kernels
[severity-fixtures]: https://github.com/StrudelDoodleS/superglm/blob/6643ed9c94497eba0efb8869b945a17c5d380caf/tests/_severity_fixtures.py
[severity-tests]: https://github.com/StrudelDoodleS/superglm/blob/6643ed9c94497eba0efb8869b945a17c5d380caf/tests/test_declared_severity_families.py
[prototype]: https://github.com/StrudelDoodleS/superglm/blob/6643ed9c94497eba0efb8869b945a17c5d380caf/notes/research/severity-family-prototype/declared_family.py
[gpd]: https://github.com/StrudelDoodleS/superglm/blob/6643ed9c94497eba0efb8869b945a17c5d380caf/src/superglm/distributional/families/generalized_pareto.py
[gg]: https://github.com/StrudelDoodleS/superglm/blob/6643ed9c94497eba0efb8869b945a17c5d380caf/src/superglm/distributional/families/generalized_gamma.py
[model]: https://github.com/StrudelDoodleS/superglm/blob/6643ed9c94497eba0efb8869b945a17c5d380caf/src/superglm/distributional/model.py
[posterior]: https://github.com/StrudelDoodleS/superglm/blob/6643ed9c94497eba0efb8869b945a17c5d380caf/src/superglm/distributional/posterior.py
[fit-state]: https://github.com/StrudelDoodleS/superglm/blob/6643ed9c94497eba0efb8869b945a17c5d380caf/src/superglm/distributional/fit_state.py
[null-model]: https://github.com/StrudelDoodleS/superglm/blob/6643ed9c94497eba0efb8869b945a17c5d380caf/src/superglm/distributional/null_model.py
[smoothing-derivatives]: https://github.com/StrudelDoodleS/superglm/blob/6643ed9c94497eba0efb8869b945a17c5d380caf/src/superglm/distributional/smoothing/derivatives.py
[response-bounds]: https://github.com/StrudelDoodleS/superglm/blob/6643ed9c94497eba0efb8869b945a17c5d380caf/src/superglm/distributional/families/_response_bounds.py
[weights]: https://github.com/StrudelDoodleS/superglm/blob/6643ed9c94497eba0efb8869b945a17c5d380caf/src/superglm/distributional/weights.py
[pyproject]: https://github.com/StrudelDoodleS/superglm/blob/6643ed9c94497eba0efb8869b945a17c5d380caf/pyproject.toml
[older-severity]: https://github.com/StrudelDoodleS/superglm/blob/6643ed9c94497eba0efb8869b945a17c5d380caf/notes/research/2026-10-31-severity-family-framework.md
[older-tail]: https://github.com/StrudelDoodleS/superglm/blob/6643ed9c94497eba0efb8869b945a17c5d380caf/notes/research/2026-10-31-large-loss-pot-sweep.md
