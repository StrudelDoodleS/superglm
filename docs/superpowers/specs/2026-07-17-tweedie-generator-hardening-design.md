# Tweedie Compound Poisson–Gamma Generator Hardening

**Status:** Approved public contract on 2026-07-17

## Context

`generate_tweedie_cpg()` implements the compound Poisson–Gamma representation of a
Tweedie random variable for `1 < p < 2`. Its formulas are correct and are already
covered by mean, variance, and zero-mass tests. The current implementation, however,
validates none of its inputs. It documents scalar `phi` while incidentally accepting a
length-`n` vector; the official Tweedie profile notebook relies on that vector behavior
to generate prior-weighted observations with `phi / w`.

This change formalizes that useful behavior and rejects ambiguous or numerically
unrepresentable inputs before they can silently generate the wrong distribution.

## Goals

- Preserve scalar-generator results and RNG reproducibility for existing valid calls
  using ordinary Python `float` or NumPy `float64` inputs.
- Formally support per-observation dispersion for weighted simulations.
- Reject invalid shapes, domains, complex values, NaNs, infinities, and derived
  parameters that NumPy cannot represent safely.
- Never misclassify an underflowed positive-claim Gamma draw as a structural Tweedie
  zero.
- Return a finite, non-negative `float64` vector of exact shape `(n,)`.
- Keep the implementation limited to the compound Poisson–Gamma regime `1 < p < 2`.

## Non-goals

- General multidimensional NumPy broadcasting.
- Tweedie generation for `p <= 1` or `p >= 2`.
- Changing density evaluation, profile likelihood, fitting, or prior-weight semantics.
- Replacing NumPy's random-number algorithms or adding another simulation backend.
- Refactoring the wider test suite as part of this branch.

## Public contract

`generate_tweedie_cpg(n, mu, phi, p, rng=None)` accepts:

- `n`: a non-negative integer accepted through the integer-index protocol; booleans are
  rejected.
- `p`: one finite real scalar satisfying `1 < p < 2`.
- `mu`: either one finite, strictly positive real scalar or a one-dimensional real array
  of exact shape `(n,)`.
- `phi`: either one finite, strictly positive real scalar or a one-dimensional real array
  of exact shape `(n,)`.
- `rng`: the existing NumPy `Generator`-compatible object, or `None` to construct a
  default generator. A compatible object must expose callable `poisson` and `gamma`
  methods; its calling convention is otherwise unchanged.

Python scalars and zero-dimensional NumPy arrays count as scalars. A one-element vector
does not count as a scalar when `n != 1`. Arrays such as `(n, 1)`, `(1, n)`, and other
broadcastable multidimensional shapes are rejected rather than expanded implicitly.
Booleans, complex values (even with zero imaginary part), numeric strings, and
object-dtype numeric containers are not accepted as real numeric inputs for `p`, `mu`,
or `phi`; callers must supply an ordinary real numeric dtype.

For `n == 0`, scalar `mu`/`phi` or empty vectors are valid and the function returns an
empty `float64` vector without a random draw. Scalar values and `p` still undergo their
ordinary domain checks.

The documented return value is a newly allocated, finite, non-negative `float64` array
with shape `(n,)`.

## Validation and data flow

Validation occurs before the first random draw whenever the condition is knowable from
the inputs:

1. Validate `n` and scalar `p` without accepting booleans, complex values, or non-scalars.
2. Normalize scalar or exact-vector `mu` and `phi` into owned `float64` arrays of shape
   `(n,)`; reject complex values before conversion.
3. Require all normalized values to be finite and strictly positive.
4. Compute the compound Poisson–Gamma parameters under explicit floating-point error
   handling:

   ```text
   lambda_i = mu_i ** (2 - p) / ((2 - p) * phi_i)
   alpha    = (2 - p) / (p - 1)
   beta_i   = phi_i * (p - 1) * mu_i ** (p - 1)
   ```

5. Require `lambda`, `alpha`, and `beta` to be finite and strictly positive. Preserve
   the existing direct arithmetic order for ordinary-call bit compatibility; joint
   CPG constraints imply that a directly underflowed `beta` cannot be rescued while
   retaining a NumPy-safe `lambda`. Match NumPy's exact Poisson safety boundary:

   ```text
   lambda_max = float(int64_max) - 10 * sqrt(float(int64_max))
   ```

   Reject `lambda > lambda_max`; NumPy accepts the endpoint itself.
6. Draw `N_i ~ Poisson(lambda_i)`. Require an integer, non-negative result no greater
   than the signed 64-bit Poisson output limit and of exact shape `(n,)`. Before the
   Gamma draw, require the positive-event shapes `alpha * N_i` to be finite and
   strictly positive.
7. Draw positive observations with `Gamma(alpha * N_i, scale=beta_i)`. Validate the
   raw Gamma result before assigning it into `y`: it must be real, have exact shape
   `(count_nonzero(N),)`, and be finite and strictly positive. Exact zero here is a
   numerical underflow, not a structural Tweedie zero. Validate the final output shape,
   finiteness, and non-negativity after assignment.

Invalid pre-draw inputs must not advance the supplied RNG. A failure depending on the
realized Poisson counts can occur only after that Poisson draw; no Gamma draw occurs in
that case. A zero or non-finite realized Gamma value is necessarily detected after both
draws have advanced the RNG; the function does not clip or resample because either
response would change the requested distribution.

## Errors

- Raise `TypeError` when `n` is boolean/non-integral or `rng` does not provide the
  required callable methods.
- Raise `ValueError` for invalid scalar/array shape, non-real data, non-finite values,
  domain violations, or numerically unrepresentable compound parameters.
- Preserve the original exception as the cause when a NumPy sampler rejects otherwise
  validated parameters, while raising a generator-specific `ValueError` with parameter
  context.
- Raise `ValueError` if an otherwise valid Gamma draw underflows to zero or overflows to
  a non-finite value. This can occur with NumPy itself for extreme but finite CPG
  parameters, so it is a numerical representability failure rather than proof of a
  malformed compatible object.
- Raise `RuntimeError` if a Generator-compatible object violates an expected draw shape
  or dtype, returns invalid Poisson counts, or returns negative/complex Gamma values.

Messages identify the offending argument or derived quantity and state the accepted
contract; tests assert useful message fragments rather than entire strings.

## Compatibility

- A valid scalar call using ordinary Python `float` or NumPy `float64` inputs with the
  same NumPy seed must remain bit-for-bit identical to the pre-hardening implementation.
- Lower-precision NumPy scalars and zero-dimensional NumPy arrays remain accepted, but
  are intentionally normalized before parameter arithmetic: `p` becomes a Python
  `float`, while `mu` and `phi` become `float64` values. Their compatibility guarantee
  is therefore exact output and RNG-state equivalence to those explicit normalized
  inputs, not to the pre-hardening dtype-sensitive arithmetic path.
- Existing exact shape-`(n,)` `mu` behavior remains supported.
- Existing exact shape-`(n,)` `phi` behavior becomes documented and tested, enabling
  correct weighted generation with `phi / w` in one vectorized call.
- Accidental one-element-vector and multidimensional broadcasting is intentionally
  rejected unless the vector's exact shape is `(n,)`.

## Testing

Implementation follows red-green TDD. Focused tests will cover:

- scalar and exact-vector `mu`/`phi`, including per-observation formula parameters;
- bit-for-bit Python-`float`/NumPy-`float64` scalar seeded compatibility and exact
  lower-precision/zero-dimensional equivalence to explicit normalization;
- weighted `phi / w` moments and zero probabilities;
- `n == 0`;
- invalid `n`, `p`, `mu`, and `phi` types, shapes, domains, complex values, NaNs, and
  infinities;
- validation before RNG use;
- the exact NumPy Poisson-limit endpoint and derived Gamma-shape representability;
- a real NumPy regression near `p=2` where positive-count Gamma draws otherwise
  underflow to zero and catastrophically inflate the structural-zero mass;
- a real NumPy regression near `p=1` where an unbounded Gamma realization can overflow
  even though every input and derived parameter is finite;
- malformed Generator-compatible outputs;
- the existing ordinary and near-boundary CPG moment characterizations.

After focused tests, the change requires Ruff, the comparable touched-module type check,
an independent implementation review, all non-slow tests, and the complete test suite.
