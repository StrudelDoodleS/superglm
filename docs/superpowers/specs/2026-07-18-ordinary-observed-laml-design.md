# Ordinary Observed-LAML Geometry Design

## Scope

Generalize exact, ordinary (`non-SCOP`, `discrete=False`) `fit_reml` so every
supported built-in noncanonical family/link pair uses the negative observed
log-likelihood Hessian in Wood's Laplace determinant, inverse, and smoothing-
parameter derivatives. Canonical/equal-curvature pairs retain the existing
Fisher geometry and avoid an extra data pass. The cached `discrete=True` path
remains the explicitly documented BAM-style working-weight approximation.

This change does not alter Tweedie likelihood profiling, power estimation,
distribution algorithms, shape-constrained fitting, or post-fit covariance
policy.

## Authority and invariant

Wood (2011), Sections 2 and 3 and Appendix D, defines the Laplace term with
the negative observed coefficient Hessian. Fisher and Newton weights coincide
only when the residual-curvature term vanishes. A coefficient solver may reach
the same penalized score root by Fisher scoring, but an accepted LAML state
must satisfy all of these conditions atomically:

1. the retained coefficients pass a penalized-score/KKT certificate;
2. the determinant and inverse come from the observed Hessian;
3. implicit coefficient derivatives use that same inverse;
4. derivatives of the determinant use derivatives of the same observed rows;
5. no numerical finite-difference derivative is represented as exact.

For `u = dmu/deta`, `v = d2mu/deta2`, variance `V`, variance derivative
`V1`, and residual `r = y - mu`, the unit-dispersion observed row is

```text
Wobs = sample_weight * (u^2 / V + r * Q)
Q    = u^2 * V1 / V^2 - v / V.
```

The first derivative uses `t = d3mu/deta3` and `V2`; the exact second
derivative uses `s = d4mu/deta4` and `V3`. The implementation will evaluate
the algebra locally in `observed_geometry.py`, with closed forms for common
log-link pairs, so Tweedie distribution/profile files remain untouched.

## Curvature classification

Classification uses exact built-in types, not `isinstance` inheritance.
Fisher geometry is valid for these analytically equal pairs:

- Gaussian with identity or power 1;
- Poisson with log;
- Binomial with logit;
- Gamma with inverse or power -1;
- NB2 with a matching canonical negative-binomial link;
- Tweedie power `p` with power link `1 - p`.

Every other exact built-in family/link pairing is observed. A custom family or
link must define `reml_curvature(counterpart)` returning exactly `"fisher"` or
`"observed"`; absent or conflicting declarations fail before fitting. Custom
observed pairs must additionally provide the derivative methods required by
the requested correction order.

The existing SCOP classifier keeps its separate `scop_reml_curvature`
protocol but shares the built-in equality proof.

## Exact derivative capability

Order zero requires inverse-link derivatives through order two and variance
derivatives through order one. Order one requires inverse-link derivatives
through order three and variance derivatives through order two. Order two
requires inverse-link derivatives through order four and variance derivatives
through order three.

All built-in links receive local exact fourth-derivative support. All built-in
variance functions have a local exact third derivative: zero for Gaussian,
Poisson, Gamma, Binomial, and NB2, and
`p * (p - 1) * (p - 2) * mu ** (p - 3)` for Tweedie. Custom objects may expose
`deriv4_inverse` and `variance_third_derivative`. Missing capability raises a
precise `NotImplementedError` before the first coefficient fit.

Common positive log-link rows use allocation-light closed forms:

- Gamma: `W = w*y/mu`, `W' = -W`, `W'' = W`;
- NB2 size `theta`: `W = w*theta*mu*(theta+y)/(theta+mu)^2`;
- Tweedie power `p`: `W = w*mu^(1-p)*((2-p)*mu + (p-1)*y)`.

The NB2 and Tweedie first and second derivatives are obtained by direct
differentiation of these expressions. Other combinations use the general
analytic formula. Signed observed rows remain supported by the stable signed
centering path; the complete penalized Hessian and intercept curvature must
still define a valid local Laplace mode.

## Runtime routing and performance

`optimize_direct_reml` classifies curvature only after its early
`discrete=True` delegation. Canonical fits therefore retain their current
runtime and allocations. Exact noncanonical fits use the existing shared
`TabmatCenteringState`, so nonnegative observed rows continue through native
centered matrix kernels. Signed rows use bounded compensated chunks.

Observed row values and requested derivatives are computed as one bundle per
candidate rather than recomputing link and variance derivatives in three
separate passes. Line-search trials request order zero and skip inverse
construction. Accepted candidates request the configured derivative order.

The exact observed path intentionally pays one additional curvature pass per
candidate. That cost is necessary to optimize Wood's criterion; it is not paid
by canonical or discrete fits.

## Error handling

- Custom curvature without an explicit protocol fails before fitting.
- Missing derivative order fails before fitting, naming the missing methods.
- Non-finite rows or derivatives fail at construction.
- Nonpositive observed intercept curvature fails.
- Materially indefinite total penalized curvature fails as a non-Laplace mode.
- There is no Fisher or finite-difference fallback presented as exact LAML.

## Test design

Tests proceed in red-green cycles and cover:

1. built-in canonical/observed classification and custom protocol failures;
2. row curvature against finite differences of per-row negative likelihood for
   NB2/log, Tweedie/log, Binomial probit/cloglog/cauchit, Poisson sqrt/identity,
   and Gamma identity/log;
3. first and second observed-row derivatives against high-accuracy centered
   finite differences for every representative pair;
4. augmented dense Hessian oracles, including signed rows;
5. refitted LAML finite-difference gradients for every representative family;
6. exact order-two LAML Hessian finite differences for representative positive
   and signed noncanonical pairs;
7. full `fit_reml` routing for noncanonical pairs and no redundant observed
   pass for canonical pairs;
8. `discrete=True` explicitly bypassing ordinary observed geometry;
9. custom derivative capability failures occurring before PIRLS;
10. focused and broad REML regression suites, Ruff, mypy, and diff checks.

## Alternatives considered

The first alternative was to generalize order zero and one but reject order
two outside Gamma/log. It is smaller, but leaves a public exact-order option
arbitrarily unavailable even though every built-in derivative is analytic.

The second alternative was to enable only pairs whose observed rows are always
nonnegative. That preserves simple PSD sandwiches but is not Wood's criterion
for valid noncanonical models with signed Newton rows, including ordinary
Binomial alternatives.

The selected design implements the full analytic built-in capability while
retaining explicit failures for genuinely unsupported custom objects.
