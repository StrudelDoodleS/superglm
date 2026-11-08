# SuperLSS scalar observations and Dirichlet: implementation design

**Status:** implementation handoff, 2026-11-08. The user authorized censoring/truncation for all remaining scalar families and production Dirichlet for strictly positive continuous compositions with known components summing to one. This supersedes the D1 feasibility-only scope in the [archived family roadmap](2026-11-08-superlss-family-roadmap.md), sections 6–7. Interfaces and numerical routes below are **engineering recommendations**, not additional requirements attributed to the user. No further scope approval is needed.

**Verified source:** `1b63f7c59e3f351fcb194f0eb25ac4e1106f2ca7`, initially clean, branch `t3/superlss-dirichlet-censoring-design`, worktree `/home/max/.t3/worktrees/superglm/t3-2328ce62`. This checkout was supplied; no checkout was created, switched or rebased. The [previous validation](../../benchmarks/declared_scalar_families_validation.md) belongs to implementation source `6e5639a115e0def2f8b1d213fed3b96eed8eb738`; the required base adds documentation. Its full baseline was not rerun. This document and its [implementation plan](2026-11-08-superlss-observations-dirichlet-plan.md) are the only intended changes in this stage.

## 1. Delivery and architectural decisions

Deliver all four remaining scalar adapters—`TwoPieceNormalLSS`, both parametrisations of `TwoPieceLogNormalLSS`, `NegativeBinomialLS`, `TweedieLSS`—with the existing observation API and working likelihood/score/Hessian fits. Audit all eleven scalar families together. Deliver a public `DirichletLSS`, including fixed-penalty and smoothing fits, prediction, joint inference, scoring/CV, retention and serialization. This is implementation work after this design, not another feasibility project.

Reuse the existing family/plan boundary and coefficient/smoothing solvers. Parameter count is already generic; response rank is not. Keep the immutable declared **scalar** adapter scalar. Dirichlet is a separate named adapter using the same `DistributionalFamily` protocol. Do not add family-name branches to solvers.

The alternatives considered were:

| Choice | Selected route and reason | Alternative and limitation |
| --- | --- | --- |
| Scalar observation derivatives | Analytic normal pieces, differentiated NB series/beta fraction, differentiated Tweedie Poisson/gamma mixture | Differencing CDF values loses tails and multiplies fit cost; per-row SciPy integration is unsuitable as the default |
| Higher bounded derivatives | Ship core orders 0–2 first for these four; refuse optional bounded analytic directions explicitly | Generating third/fourth tails now adds substantial numerical work and two-piece regularity hazards without being necessary for coefficient fitting |
| Dirichlet predictors | One log-alpha predictor per named component | Mean/precision uses coupled transforms and different penalties/model restrictions; it is a separate future parametrisation |
| Vector support | Small structural response metadata and shared shape validation | Globally accepting any 2-D array would weaken scalar validation; coercing columns to separate observations changes the law |

No general interval censoring, public arbitrary-family factory, beta family prerequisite, new optimizer, dependency, version or release change. Dirichlet counts, unknown totals, structural/rounded zeros, incomplete reserving triangles and component censoring are separate laws. No pseudocounts or automatic repair of materially invalid compositions.

## 2. Source audit: existing reusable pieces and actual gaps

Paths here are relative to `src/superglm/` unless otherwise stated.

| Existing code at the pin | Implementation consequence |
| --- | --- |
| `distributional/family.py`: `ParameterSpec`, `NaturalLikelihoodEvaluation`, `ResponseWindowFamily`, optional curvature protocols | Keep one log likelihood per row, score `(n,K)`, raw signed upper-packed Hessian `(n,K*(K+1)//2)`; `max_derivative_order=2` describes core evaluation, not optional directions |
| `distributional/families/_response_bounds.py`: `bounded_channels`, `censored_carrier`, `bounded_higher_derivatives`, `refuse_bounded` | One owner for observation accounting; frequency multiplies the complete normalized contribution once |
| `distributional/observation.py`: `(lower,upper]`, immutable bounds, positional slicing | `on_support(0)` currently erases a lower bound equal to zero: correct for continuous mass integration, wrong for NB/Tweedie atoms. Exact-row validation currently permits `y==lower`; enforce the documented open lower endpoint |
| `kernels/_response_tails.py`, `_declared_bounds.py` | Reuse stable log differences, normal windows and narrowly admitted quadrature. A finite CDF value is not derivative authority |
| `kernels/_incomplete_gamma.py` | Existing compiled shape/argument jets, small-shape route, scaled argument derivatives, fixed-`x/a` coordinates, explicit convergence failure; do not invent a second gamma special-function engine |
| `families/tweedie.py`: `_compound_poisson_parameters`, `_series_window`, `_SeriesBlock`, CDF/quantile/atom/prior helpers | Correct mixture and row-law mapping exist. Its `1e-12` **absolute CDF** Poisson tail cutoff is insufficient for a tiny log window or its derivatives |
| `families/negative_binomial.py`, `kernels/negative_binomial.py` | Exact count mapping, carriers, finite-NB2 range and Poisson-limit disclosures exist. Bind/evaluate currently require complete rows; censored noninteger thresholds must not be validated as exact counts |
| `families/two_piece.py` | Existing mean-to-location loading and join checks must be reused. Plan identity does not hash bounds, and `take()` currently carries the unsliced complete observation; both need updating before bounded plans exist |
| `families/tweedie.py:predictor_rows` | Fast compiled complete-row path must return `None` on a bounded plan until an observation-aware fast path is implemented; otherwise it bypasses the new likelihood |
| `residuals.py` | Bounded atom residuals are explicitly unavailable today. Keep that honest refusal in this delivery; joint log scoring is available. NB still has no CDF/quantile capability |
| `_likelihood_cache.py` | Cached array authority assumes 1-D immutable buffers. Do not silently certify 2-D response reuse; Dirichlet initially uses the uncached generic route |

Historical roadmap counts of nine families and five bounded families are stale. The current scalar count is eleven, seven already bounded. Update public documentation at integration; leave archived evidence historically accurate.

## 3. Shared observation foundation

### Law, endpoints and row provenance

For unit replication mass, use exactly:

```
exact: log f(y)                 - log P(lo < Y <= hi)
right: log P(c < Y <= hi)        - log P(lo < Y <= hi)
left:  log P(lo < Y <= c)        - log P(lo < Y <= hi)
```

Frequency `m` multiplies this entire expression and all its derivatives once. A prior weight changes the distribution first; it is not an additional frequency multiplier. Censored rows have zero density carrier. An observed window equal to its truncation window is identically zero, including every derivative. An exact NB observation in a singleton count window is also identically zero, including its factorial carrier.

Keep the public `truncation=(lower,upper)` and `censored=False/True/'none'/'left'/'right'` forms. Continue shape-checking before selecting positive-weight rows, then validate values on retained rows only. Errors name original input positions. Preserve root/take provenance, duplicate/permuted take maps and immutable owned arrays.

Extend `ResponseBounds.on_support(support_lower, *, atom_at_lower=False)` and `bind_observation(..., atom_at_lower=False)`; existing continuous callers retain their integration behavior. For an atom, erase only bounds **strictly below** support; a bound at support excludes the atom. Do not call continuous `log_window_values(..., support_lower=0)` for atomic laws: it also collapses the distinction. Add a shared atomic-aware value dispatcher or pass the same endpoint policy explicitly.

General response-bound validation enforces `lo < y <= hi` for exact rows. Censored thresholds are finite limits, not exact realizations: retain the existing within-window requirement, and check actual nonempty **support intersection** in the family binder. A right threshold at zero can be meaningful for a strictly positive continuous law; a left threshold at zero can be meaningful for an atomic law. Separate exact-support validation from threshold validation, including the seven existing adapters. Do not evaluate a log density/carrier at an invalid censored placeholder. Evaluate exact subsets, or pass a family-owned harmless density placeholder only internally; always retain/hash the original threshold.

Binding rejects data-impossible windows (NB empty count sets, atom excluded from a zero-only window). A mathematically positive but numerically unresolved probability at a *trial parameter state* is a numerical evaluation failure, not malformed data and not zero probability.

### Numerical interface, normalization and authority

Keep `Jet=(J0,J1,J2,...)` and canonical packed order. New core window functions accept `order=0,1,2` and return unit-mass **natural-coordinate** jets; intermediate log coordinates are explicitly transformed. Return NaNs internally for an unresolved row and translate them into existing finite placeholders plus `valid=False`, or the existing numerical-evaluation exception understood by the trial controller. Public probability queries raise a useful range/convergence error instead of returning a guessed finite answer. No clipping a probability to an arbitrary minimum.

Add one core positive-sum helper in `_response_tails.py`. For component log values `l_t`, scores `g_t`, Hessians `H_t`, and normalized positive shares `v_t`:

```
L = logsumexp(l_t)
g = sum(v_t*g_t)
H = sum(v_t*H_t) + sum(v_t*(g_t-g)*(g_t-g).T)
```

Use streaming scaled sums and centered score moments; no full terms-by-rows-by-parameters tensor. This is used by two-piece split windows, NB finite sums and the Tweedie mixture. It is a core-order helper, not a new general AD framework.

Use direct smaller tails and `log_window_jets` for well-separated endpoints. A narrow window uses a law-specific endpoint/short-window formula or existing bounded-order quadrature with an admission bound on density **and requested derivative** variation. Merely observing agreement of two quadrature orders is an error estimate, not a certificate. Windows touching a singular support endpoint use the tail engine, not a smooth-density quadrature assumption.

Combined normalization needs a second check: accurate numerator/denominator jets can cancel. Compute identical events algebraically. For narrow/strongly conditioned rows, center component log densities and scores on the same anchor before forming numerator and denominator. For a finite conditional sum this computes `log sum exp(l_t-l_anchor)` and score differences directly, avoiding subtraction of large common channels. Extend `bounded_channels` with an optional **unit-mass normalized-row override**, supplied only for selected rows; exclude those rows from the ordinary subtraction path. Its shared implementation applies replication once; family callbacks supply numerical channels, not new observation parsing.

For the four new **bounded** adapters, keep the full normalized contribution in the optimizing channel and use a zero carrier on every row of that bounded plan. Complete plans retain their existing carrier/identity. This is essential for NB: computing `small_conditional_loglik - large_factorial_carrier` and later adding that carrier loses the answer. Bounded NB exact rows therefore need a stable full log-PMF seed, not addition of the old two large channels. The override requires zero carrier and writes the complete contribution directly. Censored density carriers are always zero; both representations of ordinary exact rows must give the same full density. If the centered route still cannot resolve user-visible derivatives, decline that trial explicitly. Do not add a global shape cutoff.

Initially every new bounded optional first/second curvature-direction method, including certified variants, calls `refuse_bounded`. Preserve complete-row methods. Existing smoothing fallback may difference the **analytic bounded Hessian** with its step/refinement evidence; this is distinct from uncontrolled runtime finite differences of CDFs or density derivatives. Failed direction evidence leaves the retained fit and its convergence/uncertainty disclosures intact. Bounded Fisher information remains unavailable; selecting it explicitly is refused. Core order-zero/one/two values must agree wherever order two is admitted.

## 4. Two-piece observations

Use the shipping epsilon-skew convention: `T=location+scale*W`, left scale `s-=1-skew`, right scale `s+=1+skew`, density `phi(z)/scale`, `z=(T-location)/(scale*s_side)`. Its side masses are `s-/2` and `s+/2`. The generic two-piece density/CDF construction is given in [Rubio and Steel, section 2](https://arxiv.org/pdf/1406.7674); the following endpoint formulas and regularity conclusions are deductions for this repository's convention.

For `t<=location`, `F(t)=s-*Phi(z-)`; for `t>=location`, `S(t)=s+*Phi(-z+)`. A same-side finite window is `s_side` times the corresponding normal window. A crossing window is the positive sum

```
s- * [Phi(0)-Phi(z_lo)] + s+ * [Phi(z_hi)-Phi(0)].
```

Differentiate these formulas analytically through `(location,scale,skew)`. Reuse normal tail/Mills primitives, normal narrow-window admission and the shared sum helper. **Do not** feed a clipped bound `location` to a normal-window derivative routine that treats endpoints as constants: the split point moves with parameters. At the split, the standardized endpoint is the constant zero; differentiate the explicit half-tail expression. This supplies correct crossing and join-endpoint core jets.

For `TwoPieceLogNormalLSS`, transform positive endpoints with `log`, map nonpositive lower limits to `-inf`, and reject zero-probability upper limits. Carry close endpoint widths with `log1p((hi-lo)/lo)`, not `log(hi)-log(lo)`. The `-log(y)` Jacobian belongs only to exact densities. In mean parametrisation use the existing `location=log(mean)-log_mean_loading(scale,skew)` and its analytic first/second derivatives; every cross derivative of this map is required. Preserve scale-floor and skew-bound settings.

The exact log density is C1 with a curvature jump at a movable join. A tail probability is C2 but generally not C3 there; censoring does not make unrestricted fourth derivatives legitimate. Keep `continuous_curvature=False` and the EFS default. Deliver bounded core jets at join endpoints, and explicit optional-direction refusal for all bounded plans in the first implementation. Complete-row join handling remains unchanged.

Start bounded fits from exact rows where informative, otherwise deterministic finite window/threshold representatives in transformed coordinates. Starts may be heuristic; the objective must never use imputed exact responses. All-censored or weakly identified data may lack a finite optimum; retain normal convergence/separation disclosure. Test both signs of skew and its configured near-wall region; do not choose the branch from skew's sign.

## 5. Negative-binomial observations

### Count law and endpoint conversion

At frequency semantics the count law has mean `M=mean`, size `R=theta`, mass multiplier `m=w`. At prior semantics, `N` has mean `M=w*mean`, size `R=w*theta`, response is the recorded rate `N/w`, and `m=1`. Reuse `_certified_count_mapping` for **exact rows only**; its finite exact-count range remains `[0,2**53)`. Censored thresholds need not be integers. Keep the existing finite-NB2/Poisson-limit disclosures.

Frequency windows contain integers `L=max(0,floor(lo)+1)` through `U=floor(hi)` inclusive, with explicit infinite endpoints. Thus `(0,1]` contains just 1, `(-inf,0]` contains just 0, `(0.1,0.9]` is empty, and right censoring at integer `c` starts at `c+1`.

For prior rates, preserve the existing binary64 recorded-rate convention: define `h(c,w)` as the largest count whose **recorded float64 value** `k/w <= c`, with `h=-1` for negative `c`. Locate it from `floor(w*c)` and correct using comparisons of neighboring `k/w`; handle multiplication overflow by exponent scaling, infinities separately. Require relevant finite boundary counts to lie in the existing distinguishable exact-count range and adjacent support points to be distinguishable, as the exact-count mapper does; decline ambiguous range cases. Use `L=h(lo,w)+1`, `U=h(hi,w)`. The NB law still has infinite count support: an infinite upper endpoint stays infinite, and the `2**53` representation restriction must never truncate its probability mass. Apply the same finite-boundary restriction to frequency windows. This is an explicit extension of the existing count encoding, not tolerance-based rounding of arbitrary censor limits. In ordinary nonboundary cases it is `floor(w*c)`; at `c=float(k/w)` it includes `k` on the closed upper side and excludes it on the open lower side. Tests use non-power-of-two exposures and `nextafter` neighbors. Document this convention in the public observation guide.

### Probability and derivative algorithm

With `p=R/(R+M)`, `q=M/(R+M)`,

```
P(N=k) = Gamma(k+R)/(Gamma(R)*Gamma(k+1)) * p**R*q**k
P(N<=k) = I_p(R,k+1)
P(N>k)  = I_q(k+1,R).
```

The PMF mapping agrees with [SciPy 1.15.3 nbinom](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.stats.nbinom.html). The incomplete-beta identity follows by summing that PMF; it is not a regression-derivative service supplied by SciPy.

1. **Singleton/small finite windows:** log-PMF seed plus recurrence `p_(k+1)/p_k=q*(k+R)/(k+1)`, accumulated around the largest term. Propagate its analytic two-coordinate jets; use compensated harmonic differences and a full-density stable seed rather than subtracting huge gammalns/carriers. For `k>0`, `t=R+k`, the seed is `-t*D((R/t,k/t)||(p,q)) + gnorm(R)+gnorm(k)-gnorm(t)-log(k)`. For `k=0` it is `R*log(p)`. Evaluate the two-component deviance with scaled/compensated `log1pmx` formulas, not direct large entropy subtraction; analytically compose the same jets. The same recurrence forms centered conditional derivatives. A singleton exact conditional law is zero without numerical subtraction.
2. **Wide/one-sided windows:** new order-two incomplete-beta tail primitive, using the fraction in [DLMF 8.17.22–23](https://dlmf.nist.gov/8.17#v), symmetry and a log prefactor. Propagate Taylor jets through the recurrence in `(a,b,logit(x))`, using stable `log(x)`/`log(1-x)`; compose `a=R,b=k+1,logit(p)=log(R)-log(M)` or the symmetric upper tail. Natural-parameter conversion occurs only after stable log-coordinate computation. With `t=a+b`, its log prefactor is `-t*D((a/t,b/t)||(x,1-x))+gnorm(a)+gnorm(b)-gnorm(t)-log(a)`, using the same stable two-component deviance as the PMF seed. Do not subtract huge gammalns. An integer shape can terminate the value recurrence without terminating its shape derivatives; a zero primal coefficient is not a derivative stopping condition.
3. **Admission/fallback:** fraction convergence checks every requested coefficient, not just its value, with scaled changes over successive iterations and an iteration cap (initial engineering budget 100,000). Compare independent recurrence and fraction on overlap. A tail difference whose error amplification consumes the core tolerance falls back to the direct count sum. One-sided sums stop using an omitted-tail bound, not a fixed number of tiny terms. For an upper sum beyond its mode use `rho=max(q,q*(k+R)/(k+1))<1`; omitted PMFs are geometrically dominated. Bound first/second derivatives with polynomial envelopes in count, using `psi(k+R)-psi(R)<=k/R` and `|psi1(k+R)-psi1(R)|<=k/R**2`, and sum the resulting geometric moments through degree two. Lower sums terminate at zero.
4. **Range:** retain existing complete-density restrictions. For windows add a work limit measured in recurrence/fraction cells, not a new size wall; start with the existing aggregate NB recurrence budget `200,000,000` cells per evaluation. The beta fraction is an empirically validated analytic route, not a rigorous all-domain derivative certificate. Outside its tested/admitted envelope try the bounded recurrence; if neither resolves the requested channels within budget, decline the state with row and route details.

The existing rowwise Poisson surrogate is not automatically uniform in a tail window. Do not classify a bounded row from its recorded censor limit alone, or zero size derivatives just because a complete density would. Use finite NB beta/recurrence for the **exact numerator as well as the denominator** unless the whole normalized row's derivative error validates the same limiting approximation. Add a private finite-law policy/entry point rather than silently invoking the complete kernel's surrogate. Pure Poisson window jets from the gamma primitive are an optional acceleration only after that check. No discontinuous density/window target mismatch is acceptable. Initialize from exact rows or deterministic admissible count representatives only for starts; never apply the exact-count mapper to a censored threshold.

`NegativeBinomialLS` gains `ResponseWindowFamily` and `censored_response=True`; CDF/quantile/random remain false. A window evaluator does not require a new public quantile implementation.

## 6. Tweedie observations

### Mixture and core jets

For `1<p<2`, reuse

```
lambda = w*mean**(2-p)/(dispersion*(2-p))
a      = (2-p)/(p-1)
b      = dispersion*(p-1)*mean**(p-1)/w
```

where `w` is the prior weight, otherwise one. `N~Poisson(lambda)`, `Y|N=j ~ Gamma(j*a,scale=b)` for `j>=1`, and `Y=0` for `j=0`. This representation and zero mass are in [Dunn and Smyth (2005), section 3](https://gksmyth.github.io/pubs/tweediepdf-series-preprint.pdf). The **window derivatives and error strategy below are this design's deductions**, not claims that the density paper supplies them.

```
P(lo<Y<=hi) = 1{lo<0<=hi}*exp(-lambda)
             + sum(j>=1) Poisson(j;lambda)*P(lo<G_(j*a,b)<=hi).
```

The atom term is included iff `lo<0<=hi`; `(0,hi]` excludes it. Left censoring at zero can be atom-only, right censoring at zero selects the positive part. Whole support, zero-only and empty events have explicit branches. Do not turn `0` into `-inf` using continuous support clipping.

Build gamma window jets in `(shape,log(scale))` on `_incomplete_gamma` and the existing gamma window infrastructure. `log_gamma_tail_scaled_into` returns **plain partials multiplied by argument powers**. `log_gamma_tail_jet_into` returns **factorial-divided Taylor coefficients**; `along_ratio=True` varies `x=(a+da)*(x/a)*(1+e)`. These contracts are different: test composition and factorial conversion directly. For `x=endpoint/b`, chain through shape `j*a` and scale. Near `x/shape=1`, use fixed-ratio coordinates and the shipping stable prefactor. Positive narrow windows use log-response widths and admitted derivative quadrature; zero-touching windows use tails.

Add Poisson log jets and gamma log-window jets, then use the shared positive-sum helper. Differentiate in convenient intermediate coordinates `(log(lambda),a,log(b))`, then analytically compose with `(mean,dispersion,power)` including second derivatives and the prior weight. Form powers/rates/scales on logs with exponent-scaled products. No parameter perturbations, per-row SciPy quadrature or mandatory symbolic runtime.

### Omitted mass and derivative admission

Reuse the existing Poisson window as an initial term range, **not a stopping certificate**. Grow it until omitted mass and derivatives are small relative to the accumulated *window* probability. This must work when the window is far rarer than `1e-12`.

A concrete conservative remainder is available without evaluating all omitted gamma windows. For positive component `j`, let `T=Y/b~Gamma(j*a,1)` and intermediate coordinates `z=(log(lambda),a,log(b))`. The joint component score and Hessian are

```
S = (j-lambda, j*(log(T)-psi(j*a)), T-j*a)
H_LL=-lambda; H_aa=-j*j*psi1(j*a); H_aB=-j; H_BB=-T
```

with other Hessian entries zero. Over an omitted Poisson set `O`, form

```
M0 = sum_O P_j
V_L = sum_O P_j*(j-lambda)**2
V_a <= sum_{O,j>=1} P_j*(j/a + 1/a**2)
V_B = sum_O P_j*j*a.
```

Then omitted absolute first derivatives are bounded by `sqrt(M0*V_i)`. Omitted raw second derivatives are bounded by `sum_O P_j*E|H_ik| + sqrt(V_i*V_k)`. The Hessian terms use `lambda*M0`, `V_a`, `sum P_j*j` and `V_B`. These bounds follow from Cauchy–Schwarz, gamma score moments, and `psi1(x)<=1/x+1/x**2`; treat `j=0` separately. They remain bounds after restricting to any response window. Poisson tail moments through degree two reduce to shifted tails using factorial moments. Bound those shifted tails with Chernoff/geometric upper bounds, or gamma tails with justified error inflation, on the log scale so a tiny tail never rounds to a false zero certificate. Do not subtract nearly equal raw moments to form `V_L`: sum centered positive terms or use the conservative positive bound `M2+2*lambda*M1+lambda**2*M0`. Inflate actual float64 bound arithmetic outward in `u`, not a single final `nextafter`.

Propagate raw remainder bounds through log, score and Hessian and through the final coordinate map. Include gamma-kernel rounding/admission error; a Boolean 'converged' from that engine alone is not an error estimate. Use its documented measured order-0–2 envelope as numerical evidence, and independent oracle grids to extend it where required; do not relabel it a rigorous universal bound. Initial target for admitted ordinary rows: relative probability/remainder contribution at most `1e-12`, scaled predictor score/Hessian error at most `1e-10`, with scale/conditioning and downstream fit checks recorded. These are numerical admission targets, not new optimizer tolerances.

Retain the existing `20,000` Poisson upper-index budget initially, but grow beyond the old absolute-CDF cutoff within that budget. Count actual gamma work and cap aggregate work as well; select the aggregate cap from short complete-fit measurements, record it, and do not invent a parameter wall. If a deep tail, extreme rate/shape or narrow conditional normalization cannot meet the error target, return a numerical refusal. A value-only evaluation may succeed where derivatives cannot; it must never reject a point with valid order-two channels. No normal/saddlepoint approximation silently replaces the exact observation law.

Process row/term tiles, reuse the union of numerator/denominator endpoint gamma evaluations **within one call**, and release work afterward. No persistent parameter-dependent CDF cache. Complexity is proportional to admitted terms times gamma iterations, which can dominate a fit; the performance harness must report these counts. Initialize bounded rows from exact observations and finite admissible representatives for censored windows, preserving atom-only zeros. Representatives affect starts only. All-censored or atom-only samples may lack a finite optimum; preserve convergence disclosure. Keep current public CDF/quantile behavior and prior-law helpers; a later migration to the new value engine needs its own parity evidence.

## 7. Public Dirichlet interface and response policy

Recommended initial API:

```python
from superglm import DirichletLSS, SuperLSS, s

family = DirichletLSS(components=("bodily_injury", "property", "expense"))
model = SuperLSS(
    family,
    family.alpha("bodily_injury", s("age", k=8)),
    family.alpha("property", "region_score"),
    family.alpha("expense"),
    weight_semantics="frequency",
)
model.fit(X, shares)             # shares: (n,3) array or labeled frame
mean_shares = model.predict(X)   # DataFrame: component columns, X's index
alpha = model.predict_parameters(X)  # DataFrame: alpha_0, alpha_1, alpha_2
```

Require an explicit ordered tuple of unique nonempty string labels, `K>=2`, rather than inferring dimension during fit. Labels may contain spaces; `ParameterSpec.name` must be an identifier, so stable parameter names are `alpha_0` through `alpha_(K-1)`. `family.alpha(component,*terms,intercept=True,link=None)` delegates to `bind_predictor`; the generic binder remains valid. Expose `family.components` and immutable `family.component_parameters` mapping labels to parameter names. Each parameter has role naming its component, positive support and default `LogLink`. Standard compatible alternate links use the existing binding rules; missing higher inverse-link derivatives use ordinary optional fallback.

The underlying model returns a read-only `(n,K)` mean matrix. The public facade wraps vector default predictions in a DataFrame with component labels and prediction-frame index; scalar families still return their existing arrays. `predict_parameters()` retains its existing natural-parameter table semantics and parameter names. Concentration is `alpha.sum(axis=1)`; it is derived, not another independently fitted column. Log-alpha changes generally affect both composition and concentration; do not describe these as isolated mean effects.

Response handling:

- Arrays are positional `(n,K)`. Labeled responses must contain each declared component exactly once; reorder by label to `family.components`, reject missing/extra/duplicate columns. Never take the last `K` columns or infer a reference category. Pandas response indices must match `X` positionally when both have indices; do not silently join/reorder rows. Index-less frame types are positional as before.
- Check shape/labels on original rows; convert and validate values only after the zero-weight take. Retain original row provenance. Reject complex/bool values, NaNs/infinities, nonpositive components, incomplete rows and material failure to sum to one.
- With `u=2**-53`, use a compensated row sum and a documented roundoff allowance `8*gamma_(K+1)` times row magnitude. Within this allowance, interpret the row as the **mathematically normalized supplied positive shares** `x_j/sum(x)`. Retain/hash supplied values on retained rows and this normalization-policy version. Use log ratios/compensated sums when evaluating that normalization, not a repeated in-place divide that drifts on replay. Outside the allowance, raise; no percentage/count conversion. This small normalization is documented, not an undisclosed material repair.
- Integer frequency weights and unit prior weights are supported. Nonunit prior weights are refused: no reproductive prior-weight Dirichlet law has been specified. Zero-weight invalid contents may be discarded as in current scalar fits; they remain represented in provenance, not in the likelihood.

Prepared normalized log shares belong to the immutable bound likelihood plan, live until that plan is discarded, and are rebuilt/sliced on changed response, component order, normalization policy or row take. They do not depend on theta, basis or penalty, so parameter/basis/penalty changes do not invalidate them. Their precision is float64, their bytes are not identity authority, and their retained memory is measured. Do not add a parameter-dependent Dirichlet cache in this delivery.

## 8. Dirichlet numerical kernel and inference

Write `A=sum(alpha)`, `mu=alpha/A`, and `y` for the mathematical normalized composition. The row log density is

```
l = lgamma(A) - sum(lgamma(alpha_j)) + sum((alpha_j-1)*log(y_j))
s_j = psi(A) - psi(alpha_j) + log(y_j)
H_jk = psi1(A) - 1{j=k}*psi1(alpha_j)
T_jkl = psi2(A) - 1{j=k=l}*psi2(alpha_j)
Q_jklm = psi3(A) - 1{j=k=l=m}*psi3(alpha_j).
```

These derivatives are deductions from the density in [SciPy's 1.15.3 Dirichlet definition](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.stats.dirichlet.html). `special.polygamma(n,x)` means the `n`th derivative of **digamma**, not of log gamma ([versioned signature](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.special.polygamma.html)). Frequency multiplies every channel once. Keep the entire stable log density in the optimizing channel and carrier zero: separating a huge `sum(log y)` carrier can reintroduce cancellation. This choice affects bookkeeping only, not the normalized density.

For log links, `score_eta_j=alpha_j*s_j` and `H_eta_jk=alpha_j*alpha_k*H_jk+1{j=k}*alpha_j*s_j`. Preserve signed Hessians; Fisher is `-H` in natural coordinates and is independent of the response. Implement `expected_information_natural` for complete plans. Observed information on predictors still includes the score-link term; replacing it by Fisher would change the existing inference contract.

Implement analytic first/second predictor-curvature directions without allocating `(n,K,K,K,K)`. Natural contractions are

```
dH_jk[v] = psi2(A)*sum(v) - 1{j=k}*psi2(alpha_j)*v_j
d2H_jk[v,w] = psi3(A)*sum(v)*sum(w)
              - 1{j=k}*psi3(alpha_j)*v_j*w_j.
```

Compose separable link derivatives through order four, including mixed second movement `g''*direction_a*direction_b`. A family-local directional evaluator can differentiate the displayed predictor Hessian with two directional Taylor variables, reusing link derivative columns. Cost/storage per direction is `O(n*K**2)`, not `O(n*K**4)`. Existing `_base.natural_curvature_second_direction` expects packed **full** third/fourth arrays; do not pass contracted arrays into it. This family-local efficient hook is compatible with the existing optional protocol. `max_derivative_order` stays two. If higher channels are unresolved, refuse the optional hook while retaining a valid core fit.

### Float64 stability and numerical refusal

Direct gammaln subtraction and `digamma(A)-digamma(alpha)+log(y)` lose meaningful digits at large concentration, especially near the fitted mean. Use direct formulas only on an admitted low-cancellation route. Reuse shipping `gnorm`, `psires_0123`, `sres_0123` and their scaled/asymptotic derivations; [DLMF 5.11](https://dlmf.nist.gov/5.11) and [5.15](https://dlmf.nist.gov/5.15) specify the underlying expansions.

An exact algebraic reformulation is

```
l = -A*D(mu||y) - gnorm(A) + sum(gnorm(alpha_j)) - sum(log(y_j))
gnorm(a) = a*log(a)-a-lgamma(a).
```

For `r_j=y_j/mu_j-1`, `A*D=sum(alpha_j*(r_j-log1p(r_j)))`, since `sum(alpha*r)=0` mathematically. Use the existing `log1pmx`/gamma-deviance style near zero. **Do not assume the floating-point linear term sums to zero.** Compute close ratios from compensated product differences using the supplied positive shares `x`:

```
r_j = [sum_k (alpha_k*x_j-alpha_j*x_k)] / [alpha_j*sum_k x_k].
```

TwoSum/TwoProduct or narrowly justified compensated reductions retain the residual when ordinary `A*x_j-alpha_j*sum(x)` cancels. Use scaled log-ratio evaluation away from equality or near `r=-1`; do not form an overflowing ratio. Keep aggregate sum rounding in the forward-error analysis. For score, combine `log(y_j/mu_j)` with `psires_0(A)-psires_0(alpha_j)`. For diagonal Hessians and higher differences factor reciprocal differences using `sum_{k!=j} alpha_k`, rather than subtracting nearly equal reciprocals/polygammas. Small alphas use recurrence/scaled residual primitives, not large-argument asymptotics.

As a natural-channel representation check, form stable log-link channels directly from `mu_j=alpha_j/A` and `sres_m(a)=a**(m+1)*psires_m(a)`:

```
score_eta_j = alpha_j*log(y_j/mu_j) + mu_j*sres_0(A)-sres_0(alpha_j)
H_eta_jj = -alpha_j*(sum_{k!=j} alpha_k/A)
            + mu_j**2*sres_1(A)-sres_1(alpha_j)+score_eta_j
H_eta_jk = alpha_j*mu_k + mu_j*mu_k*sres_1(A), j!=k.
```

Compute the log ratio with the compensated route above. Compare natural-to-linked reconstruction with these scaled values using scale-aware float64 error tolerances, as the NB kernel already does for representability. A finite natural Hessian that underflows away a user-visible linked channel is not valid merely because `isfinite` passes. This check diagnoses/refuses an unresolved representation; it does not overwrite a rank decision or assert coefficient accuracy at arbitrary concentration.

Mathematical support is all positive finite alphas and positive interior compositions. Numerical admission is separate: refuse nonrepresentable sums, natural/predictor channels or unresolved cancellation **after** scaling/stable routes. Do not clamp alpha to `[epsilon,constant]`, zero a genuine concentration direction or silently add a ridge. Record failed quantity and whether core or optional derivatives failed. Preserve last valid fitted states and convergence disclosure. Large-concentration information becomes ill-conditioned; use existing rank/covariance decisions and test stable predictions and covariance projections, rather than demanding coefficient accuracy in an unidentified direction.

Starts use weighted component means and a robust finite moment concentration from component variances, with a documented finite fallback for constant/degenerate data. A finite initial concentration is a start, not a fit wall. Support a supplied valid `initial` through the existing API. Constant compositions can have an unbounded concentration MLE; warn/disclose nonconvergence instead of claiming a finite optimum.

For the distribution's response covariance provide a separate family method `response_covariance(theta)->(n,K,K)`:

```
Cov(Y|alpha) = [diag(mu)-mu*mu.T]/(A+1).
```

Do not implement the scalar `VarianceFamily` protocol with a matrix. Prediction means use scaled positive normalization; a positive mathematical mean which underflows to zero is a disclosed numerical-range refusal, not a structural zero or pseudocount. Covariance uses stable products and is singular in the all-ones direction to rounding. This response covariance is distinct from coefficient covariance and posterior uncertainty of mean shares.

## 9. Vector boundary impact map and supported operations

Add frozen internal `ResponseSpec(shape=(), component_names=())` with a structural optional `ResponseSpecFamily.response_spec`; absent means scalar. For Dirichlet `shape=(K,)`, names are the declared components. Central helpers check `(n,*shape)` and optionally retained finiteness. This is response metadata, independent of predictor count and observation kind.

| Boundary / files | Required change or explicit unchanged contract |
| --- | --- |
| New `distributional/response.py`; `family.py` | Shape/label metadata, original-shape/retained-value helpers, vector prediction label protocol; no inferred vector permission from a 2-D input |
| `model.py`, `api.py` | Preserve response labels until canonical ordering; shape-only before weight selection; scalar bounds reject vector response before broadcasting; public vector prediction wrapping |
| `families/_base.py` | Generic row-count utility accepts validated row-first arrays; existing scalar `validated_float_response` remains strict; all scalar adapters retain 1-D tests |
| `fit_state.py`, `null_model.py` | Response may be `(n,K)`; weights/offsets remain `(n,)`, never compare weights to full `response.shape`; retained state stores response spec with scalar-compatible default. Diagnostic reconstruction allocates `(original_n,*response_shape)`, not a scalar buffer before assigning retained rows |
| `solver/solver.py`, `solver/chunks.py` | Replace explicit `(n,)` response guards using metadata; `take` slices only axis 0. Keep derivatives, assembly, factors, rank and coefficient geometry unchanged |
| `smoothing/endpoint_direction.py`, `_likelihood_cache.py` | Remove scalar response assumption in the derived-root branch; initially do not register Dirichlet in certified reuse. Test replay without reuse rather than weakening authority checks |
| `predictor.py`, `separation.py`, factor-smooth level recording | Dirichlet declares no scalar zero/one response boundaries. Empty-boundary scans must still record weightless/thin levels without applying scalar response arithmetic to a matrix |
| `checks/scores.py`, `checks/compare.py`, `distributional/cross_validation.py`, `model_selection.py` | Joint log score remains `(n,)`; shape/label-aware scoring and paired joint-log comparison; vector OOF allocation and zero-weight fallback `(n,K)`; default Dirichlet CV scoring only `nll`. Preserve labels and defer value coercion before outer CV selects weights/rows |
| `inference.py`, `posterior.py`, `prediction_design.py` | Coefficient covariance, per-alpha parameter prediction/SE, posterior coefficient/parameter draws remain generic; efficient analytic curvature replay; explicitly reject scalar default `quantity='mean'` for a vector mean |
| `serialization.py`, `likelihood_identity.py`, `rebuild.py` | Stable top-level classes, complete config and response-policy identity; pickle/joblib, clone/refit/hand-edit and compact replay. Do not hash transcendental prepared logs as exact identity |
| `fit_diagnostics.py`, API summaries | Algorithmic diagnostics/EDF/rank/log likelihood remain valid; do not label joint log density as a univariate deviance. Numerical-range and concentration-nonidentification messages stay visible |
| `residuals.py`, `surfaces.py`, `checks/calibration.py`, `editor/session.py`, `editor/lss/view.py` | Explicit unsupported-operation checks before scalar allocation. Initial vector editor/density/quantile UI and scalar calibration are declined, not populated by flattening shares |

Supported initially: `fit`, `fit_reml`, observed and complete-row Fisher coefficient policies, null fit, likelihood/AIC-style existing likelihood metrics, dense and chunked/discrete execution, compact and retained models, `predict`, `predict_parameters`, predictor/parameter effects, joint coefficient covariance, posterior coefficient/parameter draws, joint negative log score, paired joint-log model comparison, CV NLL/OOF and user-defined scalar scores. Existing smoothing-uncertainty correction is supported when derivative authority succeeds; its refusal/warnings remain unchanged otherwise. Keep the scalar `scores()` default unchanged: Dirichlet users explicitly request `which=("log",)`; do not silently drop requested CRPS. Paired comparisons require the same component set for labeled responses, or identical component order for positional arrays, and canonicalize each candidate separately before scoring.

Posterior scalar functionals use existing callables, e.g. `lambda a: a[:,j]/a.sum(axis=1)`, a sum of selected mean shares, a log ratio, or total concentration; `('parameter','alpha_j')` also works. No vector-valued return is silently accepted by the scalar `posterior_bounds` payload. Mean-composition posterior covariance can be calculated from parameter draws transformed to shares, preserving all cross-component dependence; tests compare this to delta-method projections on well-conditioned fixtures. No independent marginal coefficient draws.

Explicitly unsupported: scalar CDF/quantile/PIT/quantile residuals, QQ/worm/PIT plots, CRPS and threshold/tail scores, scalar Gini/actual-over-expected totals, scalar distribution surfaces, `posterior_predictive` (currently quantile-based), portfolio simulation and response editor views. Capabilities are `cdf=False, quantile=False, random=False, censored_response=False, response_mean=True, continuous_curvature=True, expected_information=True`. Do not expose stubs which accidentally satisfy optional scalar protocols. A caller can use `response_covariance` and scalar share posterior callables; a new multivariate diagnostic/sampling framework is not required for this release.

## 10. Acceptance evidence and implementation limits

The [plan](2026-11-08-superlss-observations-dirichlet-plan.md) assigns every acceptance item. At minimum:

- All-eleven matrix: exact/left/right, lower/upper/two-sided truncation, combined censoring/truncation, frequency replication, each supported prior law and each expected refusal. Include one-point/empty count windows, atom-only/excluded atom, support-touching limits, floating neighbors, deep tails and narrow conditional events. Independent value/score/Hessian oracles must be on the **normalized row law**.
- Dirichlet: `K=2` beta reduction; `K=3` independent intercept/regression fits; mixed alpha magnitudes, high concentration and near-boundary compositions; labels/order, permuted models/data/penalties, zero rows, row identity, null fit, fixed/smoothing fits, conditional/corrected covariance where available and retained/compact round trips. Include `K>3` to prevent a hidden three-component assumption.
- Mutation/unfixed evidence: omit truncation denominator, double mass, retain density carrier on censor rows, include excluded atom, round count limits, omit mean-to-location cross terms, lose simplex cross Hessian, flatten `(n,K)`, drop labels or accept a changed plan. Each must make a targeted test fail.
- Independent fit comparisons require stationarity at both compared endpoints; compare observable predictions, objective, score, covariance projections and rank. Optimizer success alone is insufficient. All intentionally unresolved or unbounded endpoints remain in the receipt.
- Complete fits, cold and warm separately: time, peak RSS, retained buffers/caches, outputs/SE/EDF/rank, iterations/retries/null refits, `diagnose()` and actual backend/kernel route. Profiling runs are separate. New laws have no old supported fit to time: compare a mathematically independent small optimizer for correctness and report absolute fit cost; use unchanged complete-row families and the seven existing bounded families as same-session baseline controls for overhead.

No performance or portability claim is established by this design. Parent owns the final full runner, mandatory real data, supported-version/native platform checks and fresh independent review. Core support for none of the four scalar families or Dirichlet may be quietly dropped if a numerical gate fails; the stage must repair the route or report the exact remaining implementation blocker. Optional higher bounded derivatives can remain explicitly unavailable as designed.

## 11. Reading and small-probe record

Read repository `AGENTS.md`, roadmap direction, archived family/framework documents, previous validation and the source/test paths named above. No production/tests/dependencies were edited, no environment was installed and no baseline suite/fit benchmark was run.

Primary-source reading on 2026-11-08:

| Source | Actually inspected | Used for |
| --- | --- | --- |
| [Maier (2014), original WU Report 125](https://research.wu.ac.at/ws/portalfiles/portal/17761231/Report125.pdf) | Abstract, model discussion introducing common log-alpha and alternative parametrisations; not every example/appendix | Regression parametrisation precedent; application details remain this design's recommendations |
| [Rubio and Steel, original two-piece manuscript](https://arxiv.org/pdf/1406.7674) | Section 2 density and CDF construction | Two-piece normalization; join derivative conclusions above are algebraic deductions |
| [Dunn and Smyth, corrected 2005 author preprint](https://gksmyth.github.io/pubs/tweediepdf-series-preprint.pdf) | Abstract, EDM setup and section 3 Poisson/gamma representation | Mixture law, atom and series limitations; new window remainder construction is not attributed to that paper |
| [Gil, Segura and Temme, original gamma-ratio paper](https://ir.cwi.nl/pub/20433/20433B.pdf) | Definitions, primary-tail choice and stable large-shape prefactor | Authority behind reuse of the shipping incomplete-gamma engine; not a claim of validated new Tweedie derivatives |
| [DLMF 8.17](https://dlmf.nist.gov/8.17), [5.11](https://dlmf.nist.gov/5.11), [5.15](https://dlmf.nist.gov/5.15) | Beta definition/fraction coefficients, gamma/polygamma expansion reference | Buildable formulas; convergence/rounding admission remains implementation work |
| [SciPy 1.15.3 Dirichlet](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.stats.dirichlet.html), [NB](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.stats.nbinom.html), [betainc](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.special.betainc.html), [gammainc](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.special.gammainc.html), [polygamma](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.special.polygamma.html) | Versioned signatures/support/formulas | Floor-compatible independent value oracles. SciPy Dirichlet takes one alpha vector; row-varying alpha needs an explicit loop in tests, and `logpdf` response orientation is component-first |

A subsecond, nonproduction probe used existing `/home/max/projects/superglm/.venv/bin/python` (Python 3.13.5, NumPy 2.4.2, SciPy 1.18.0), without importing this checkout. For `y=(.15,.35,.5), alpha=(2,4,5)`, the direct Dirichlet value matched SciPy; central differences with step `1e-4` agreed with score/Hessian to `8.2e-10`. K=2 beta values agreed; NB beta-window versus finite PMF sum differed by `1.7e-16`; a Tweedie zero-atom inclusion check differed by rounding only. A second subsecond algebra probe checked the displayed NB seed and beta prefactor identities at ordinary arguments against SciPy (maximum differences `8.0e-15` and `2.3e-15`). These sanity checks verify algebra/call semantics, **not** the proposed numerical envelope, derivative certificates or a production fit. High-precision adversarial oracles remain required implementation tests. No genuinely blocking user question remains.
