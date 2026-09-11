# Family-bound predictor API

Status: approved for implementation on 2026-09-12. Use family-bound predictor
helpers, a family-first constructor, positional predictors and explicit
specification of every family parameter. Public Tweedie helper names are `mu`,
`phi` and `p`, mapping to canonical `mean`, `dispersion` and `power`. Other
families retain their canonical parameter names as helper names.

Baseline: v0.32.0, commit `087d5983d1cba9823014c73eb8400b593a8c9217`.
The user permits breaking public API changes before 1.0; no deprecation cycle
is required for the old `SuperLSS(family=..., predictors=...)` constructor.

## Objective

Make the family own predictor names, links and ordering. Analysts describe
terms attached to columns and pass individual bound predictors to `SuperLSS`.
Keep `SuperGLM` and `SuperLSS` as the model construction entry points.
Preserve the mathematical meaning of every configuration supported by the
normalization layer. This change does not introduce new fitting algorithms,
shape constraints, full tensor smooths or disk-backed fitting.

```python
from superglm import SuperLSS, TweedieLSS, cat, s

family = TweedieLSS(power_lower=1.08, power_upper=1.92)
model = SuperLSS(
    family,
    family.mu("VehicleAge", s("Power", kind="cr", k=10), cat("Area")),
    family.phi(s("VehicleValue", kind="cr", k=6)),
    family.p(),
    discrete=True,
)
```

The explicitly empty power predictor has an estimated intercept. It is constant
across rows; it is not a caller-fixed numerical power value. Omitting a predictor
is an error.

## Approaches considered

1. Family-bound helpers passed as `*predictors`, the current preference. Each
   helper describes one named parameter, and all return the same specification
   type. The shared model constructor has a stable signature for every family.
2. One family-owned `.predictors(...)` bundle passed to `SuperLSS`. This gives
   parameter-specific keyword signatures but exposes an additional configuration
   stage. It is superseded by individual predictor helpers.
3. Per-family constructor overloads on `SuperLSS`. These can describe different
   keyword sets without creating separate estimator classes, but need editor
   evaluation and an extension strategy for custom families. Individual helpers
   avoid that dependency on constructor overloads.

Family `.model(...)` factories were rejected as the primary entry point because
they move model construction onto distributions and break the preferred common
construction style. A `predictors=[...]` collection was also considered; the
user prefers individual positional predictor specifications. Programmatically
built collections can be passed with `SuperLSS(family, *predictors)`.

Initially use small explicit helper methods sharing one binder. Generating those
methods later must cover every supported parameterization, not only default
instances. Do not replace numerical family metadata just to enable generation.

Runtime `__signature__` is not the static typing contract. IDE-specific plugins
are outside this design.

## Public constructor and configuration

The new public constructor accepts
`SuperLSS(family, *predictors, discrete=..., ...)`.
Its other execution and likelihood-weight options retain their meanings.
The family is the first positional argument, followed by predictor declarations;
execution options are keyword-only. This puts the family before its predictors
without placing ordinary positional arguments after a keyword argument. Do not
infer the family from the first predictor. The old `predictors=` keyword is an
error with a migration example. Migrate first-party examples and public
constructor tests together.

`family.mu(...)` and analogous methods produce a uniform bound predictor
specification carrying its canonical parameter name, originating family identity,
terms and predictor-level options. Helpers do not mutate the family or create a
model. Creating a specification must not read a training frame, build a design
matrix, initialize a likelihood or allocate arrays proportional to row count.

Use frozen outer specifications and defensive ownership of the specification
graph. A frozen dataclass alone does not make contained feature objects immutable.
Copy caller-supplied mutable term configuration when creating a specification.
At model construction, check all originating family identities against the
explicit family, take an owned snapshot of that family's configuration, and
validate names, links and supports against the snapshot. The originating family
reference identifies the source of a declaration; it is not itself an immutable
snapshot. Subsequent changes to caller-owned objects must not alter a constructed
model. A mutable custom family must either be successfully snapshotted at model
construction or fail clearly. Isolate each model's compilation state and preserve
shared references within an owned graph where semantically significant. Public
accessors must not expose mutable internal configuration.

`SuperLSS` lowers these specifications into its existing family and canonical
`Predictor` tuple. The internal predictor compiler, solver and inference paths
continue to consume their current representation. Keep `Predictor` as an internal
building block; it no longer needs to be constructed in normal public code.

Existing serialized numerical models should continue to load through the same
schema. Adapt the loader's construction path to the new public boundary.
Serialization records the normalized family and predictors, not a new numerical
model type.

## Names, omissions and controls

- Internal parameter identity comes from `family.parameters`. Current Tweedie
  names are `mean`, `dispersion` and `power`. Diagnostics must use the selected
  public helper spelling consistently if it differs from the canonical name.
- Positional argument order has no parameter-assignment meaning. Normalize by
  the instance's declared parameter order before entering the compiler.
- Require every predictor exactly once. `family.p()` explicitly requests
  intercept-only power; a missing power declaration is an error.
- Apply completeness to all declared parameters, without assuming a fixed arity
  or that every family has a mean parameter.
- Unknown names and names from the wrong parameterization fail at binding and
  are checked again against the family snapshot at model construction.
- Each helper accepts `*terms`. A string is one numeric column declaration;
  programmatic term collections use unpacking. Do not give bare method objects
  such as `family.p` a second meaning as predictor declarations.
- `family.mu("x", s("z"), intercept=False, link="log")` supplies optional
  controls. Default links come from the actual family configuration, including
  configured bounds. Preserve support validation and refusal of invalid empty
  designs.

## Actionable constructor diagnostics

For an incomplete specification, show a short explanation followed by a generated
example of the constructor call. Insert the missing helper in canonical order and
mark that line with an ASCII arrow. For example, when mean and dispersion were
provided but power was omitted:

```text
TweedieLSS is missing a predictor for power.

SuperLSS(
    family,
    family.mu(...),
    family.phi(...),
    family.p(...),  # <--- missing predictor; add this
)
```

This is a suggested call, not a reconstruction of the caller's source or a real
family-dependent constructor signature. `family` is a placeholder variable;
ellipses stand in for terms without choosing them, including for missing
predictors. Generate the display from family metadata and validated predictor
identities. Do not inspect caller source, evaluate expressions, or render
arbitrary user object representations. Exact spacing and wording may evolve;
retain the named problem and insertion guidance.

Every missing parameter gets its own marked line. The diagnostic requires the
missing declaration without recommending terms, intercept behavior or a constant
parameter. Do not insert a declaration automatically. Examples must use the
actual public helper names. Preserve the distinction between canonical parameter
identity and helper spelling if short names are adopted.
For a custom family without a public parameter helper, render the supported
`bind_predictor(family, "parameter_name", ...)` declaration rather than suggesting
a method that does not exist.

Give separate targeted messages for duplicate declarations, wrong-family
bindings, bare methods passed without calling them, and invalid positional
argument types. Validate argument types and family ownership before presenting
completeness advice, so a wrongly bound predictor does not masquerade as a
missing valid declaration.

## Family parameters

Built-in names verified on the baseline:

| Family | Canonical predictors |
| --- | --- |
| GaussianLS | location, scale |
| GammaLS | mean, scale |
| TweedieLSS | mean, dispersion, power |
| NegativeBinomialLS | mean, theta |
| GeneralizedParetoLSS | scale, shape |
| GeneralizedGammaLSS | mean or location, scale, shape |
| LogNormalLS | mean or location, scale |
| TwoPieceLogNormalLSS | mean or location, scale, skew |
| TwoPieceNormalLSS | location, scale, skew |

## Bound terms

Every explicit term owns its source column names and its construction settings.
Normalize into the current feature and interaction objects before compilation.

| Declaration | Meaning |
| --- | --- |
| `"x"` | `Numeric()` on column x |
| `s("x", ...)` | A configured spline on x |
| `cat("area", ...)` | A configured categorical effect on area |
| `re("group", ...)` | A configured random effect on group |
| `term("x", feature_spec)` | Bind another supported or custom FeatureSpec |
| `ti("x", "z", ...)` | Existing interaction-only tensor of two declared spline parents |
| `interaction(interaction_spec, name=...)` | Bind an existing explicit interaction type |

Use explicit annotations for the common helper arguments. Preserve the existing
meaning of spline `k`, `n_knots`, `kind`, selection and penalty settings. The
generic `term` form retains access to advanced feature options without inventing
a second set of feature semantics.

There is no dtype inference. A string column declared as `"Area"` remains a
numeric declaration and must fail clearly when the data cannot satisfy it. The
error should point to `cat("Area")` when categorical encoding is intended.

In the initial normalization layer, allow one main-effect declaration per source
column per predictor, matching the current dictionary contract. Reject duplicate
or conflicting declarations; never silently overwrite one. The same column can
have different specifications in different predictors. Stable interaction names
must obey the existing namespace and collision rules.

The input term sequence is a unified declaration syntax. Build dependencies still
place main effects before interactions. Retain declaration order within the
main-effect and interaction partitions, matching existing coefficient and
summation order. Do not promise arbitrary interleaving of coefficient blocks or
a new reporting order in this change. Canonical ordering must survive
serialization.

## The tensor correction

`TensorInteraction` on this baseline is explicitly an interaction-only `ti`
surface. It removes constant and marginal main-effect directions and requires
two spline parents. Naming that wrapper `te` would promise a different model.

```python
family.mu(
    s("Power"), s("VehicleAge"), ti("Power", "VehicleAge"),
)
```

An interaction must reference declared compatible parents. Resolve references
after reading the whole list, so placing an interaction before its parent
declarations is harmless. Missing parents and numeric parents passed to `ti`
are errors. Never add missing main effects or convert a numeric main effect into
a spline to make an interaction compile.

Standalone full `te` terms, interaction-specific independent marginal types and
multiple named effects on one source column need a broader term/compiler design.
Keep those explicit future capabilities rather than treating them as aliases.

## Typing and custom families

Expose ordinary typed helper methods on built-in families, such as
`mean(*terms, intercept=True, link=None)`. Their common return type is a bound
predictor specification. Use a shared binder for term and parameter validation,
and constructor validation for completeness, family ownership and ordering.
Check helper declarations against the numerical metadata for every supported
parameterization. Dynamically installing attributes alone is not sufficient to
provide static editor completion.

GeneralizedGammaLSS, LogNormalLS and TwoPieceLogNormalLSS can change their first
parameter name according to `parametrisation`. Their static declarations can
expose both mean and location helpers, but binding must reject the helper that
does not exist in the actual instance's parameter contract and identify the
valid alternative. Do not promise that every IDE will hide the alternate
parameterization. Encoding the mode in the family's static type would be a
separate design if that stronger completion contract is required.

Keep the solver-facing DistributionalFamily protocol unchanged. Offer a public
`bind_predictor(family, name, *terms, ...)` helper for custom families; built-in
methods use the same binding machinery. Custom families may add their own
explicitly typed parameter methods. An arbitrary runtime parameter list cannot
by itself give a language server new static attributes. Include a four-parameter
custom-family regression.

The baseline has no `src/superglm/py.typed`. Add and package the marker, verify
its presence in built distributions, and test typing from an installed wheel.
This is not a promise that the existing package-wide typing backlog is resolved.
Test the new consumer-facing surface with pinned type checkers and do not infer
universal editor behavior from runtime introspection alone.

## Validation required for implementation

- Binding: canonical order, every parameter required, explicit intercept-only
  declarations, wrong and duplicate names, type errors and invalid links.
- Diagnostics: missing-parameter names and marked repair lines, multiple omissions,
  duplicate and wrong-family declarations, bare-method hints and invalid positional
  arguments. Verify repair semantics without depending on incidental whitespace.
- Ownership: mutation of original lists, feature specs and family objects; repeated
  compilation and reusing a bound specification across models.
- Term normalization: numeric versus categorical semantics, spline settings,
  duplicate columns, parent references, named interactions and rejection paths.
- Numerical parity: compare new syntax with direct internal Predictor construction
  on well-conditioned fixed-smoothing and REML fixtures. Compare compiled designs,
  penalties, predictions, convergence outcomes and covariance. Include discrete
  execution and an interaction model, with existing tolerance policy.
- Family coverage: every built-in parameterization and a custom four-parameter
  family. Validate its instance-owned links and support bounds.
- Persistence: load existing artifacts and round-trip new configurations without
  changing the represented model or canonical term order.
- Typing: valid and invalid consumer examples, family helper names and return types,
  mode-dependent helpers and packaged py.typed. A misspelled helper must be diagnosed.
- Migration: update constructor examples, public constructor tests, internal public
  call sites and loader construction. Leave backend tests using internal Predictor
  values where that is the boundary under test.
- Run the repository's applicable lint, format, lock, dependency and test checks.
  Broad validation must honor the mandatory-real-data policy when those suites run.

The anticipated advisory release impact is `release:minor`, because this introduces
a new modelling interface and intentionally changes the pre-1.0 public constructor.
No version bump or publishing action belongs to this work.

## Research and capacity work remain separate

The ten-million-row receipt demonstrates one fit's time and memory on its measured
configuration. Disk-backed inputs or caches can reduce resident ownership, but
repeated optimizer passes then incur I/O, and live coefficient matrices require
their own treatment. This API proposal does not reopen the deferred out-of-core
project or claim a whole-fit memory bound.

## References

- [PEP 692](https://peps.python.org/pep-0692/) prefers explicit public keywords for
  IDEs and documentation where possible.
- [PEP 681](https://peps.python.org/pep-0681/) describes dataclass-like transforms,
  not automatic binding of a configuration class to a family instance.
- [PEP 561](https://peps.python.org/pep-0561/) specifies packaged typing markers.
- [inspect documentation](https://docs.python.org/3.15/library/inspect.html)
  describes __signature__ handling as an implementation detail.
