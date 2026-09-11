# Family-bound predictors implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Construct models as `SuperLSS(family, *predictors, ...)` with typed family helpers, bound terms and actionable missing-predictor diagnostics.

**Architecture:** New term declarations lower into existing feature and interaction objects. Family helpers bind one named predictor to the originating family. The public constructor validates completeness and ownership, takes owned configuration, and delegates to the existing numerical representation. No solver changes are intended.

**Tech Stack:** Python 3.12+, development Python 3.13, NumPy/SciPy, pytest, Ruff, Hatchling and the existing typing tools.

**Spec:** `docs/superpowers/specs/2026-09-11-family-bound-predictor-api-design.md`. The user approved the family-first constructor and strict completeness on 2026-09-12. The accepted Tweedie examples use `mu`, `phi`, `p`; these map to canonical `mean`, `dispersion`, `power`. Other families keep canonical helper names.

## Global constraints

- Work only in `.worktrees/fresh-0.32.0`, branch `work/fresh-0.32.0`, baseline `087d5983d1cba9823014c73eb8400b593a8c9217`.
- Keep `SuperGLM` and `SuperLSS` as the model construction entry points.
- Require every predictor exactly once; argument order never assigns parameter meaning.
- Error arrows flag omissions without choosing terms or constant/intercept behavior.
- Preserve numerical parameter names, links, supports, compilation order and artifact schema.
- Use explicit numeric/categorical declarations, one main effect per source column per predictor, and existing interaction semantics.
- `ti` means the existing interaction-only tensor with declared spline parents; do not introduce `te` or shape enforcement.
- Preserve structural custom-family support, including four parameters.
- Do not change version files or publish anything. Advisory release impact: `release:minor` because the public construction interface changes.
- Use `apply_patch` for repository edits. Follow regression and numerical-test rules in AGENTS.md.

## Files and responsibilities

- `src/superglm/terms.py`: owned bound term records, typed term helpers, and normalization into current feature/interaction dictionaries.
- `src/superglm/distributional/binding.py`: bound predictors, common binder, family ownership/completeness validation, diagnostics, and internal template binding for serialization.
- `src/superglm/distributional/families/_predictors.py` and existing family modules: statically visible family helper methods delegating to binding.
- `src/superglm/distributional/api.py`: public constructor and loader integration, preserving existing runtime options.
- Package exports and `src/superglm/py.typed`: discoverable typed public names.
- Focused tests for terms/binding/construction, existing public constructor tests, numerical fixture migration, docs, benchmarks and installed-wheel typing checks.

## Task 1: Bound term declarations and normalization

**Files:** Create `src/superglm/terms.py` and `tests/test_bound_terms.py`. Modify feature input diagnostics only if needed to give explicit numeric-versus-categorical guidance, with a regression test.

**Interfaces:**
- Produce `TermInput`, covering string, bound main term and bound interaction declarations.
- Produce `s(column, *, kind="ps", k=None, n_knots=None, ...)`, mirroring the existing spline factory's supported options explicitly; `cat(column, *, ...)` and `re(column, *, ...)` mirror their existing constructors.
- Produce `term(column, spec: FeatureSpec)`, `interaction(spec, *, name=None)` and `ti(left, right, *, n_knots=None, decompose=False)`.
- Produce `normalize_terms(terms: Sequence[TermInput]) -> NormalizedTerms`, whose attributes are `features`, `interaction_specs`, `interaction_order` and whose contents are owned copies. No distributional-family imports in this module.

- [ ] Write real normalization tests before implementation. The first consumer-visible test must fail against the baseline without an import collection error:

```python
def test_bound_terms_keep_numeric_and_categorical_semantics():
    from importlib.util import find_spec
    assert find_spec("superglm.terms") is not None, "bound terms are not implemented"
    from superglm.terms import cat, normalize_terms, s
    from superglm.features import Categorical, Numeric
    result = normalize_terms(("age", s("value", kind="cr", k=6), cat("area")))
    assert tuple(result.features) == ("age", "value", "area")
    assert isinstance(result.features["age"], Numeric)
    assert isinstance(result.features["area"], Categorical)
```

- [ ] Run `uv run pytest tests/test_bound_terms.py -q`; record the missing-feature failure.
- [ ] Implement owned declarations and normalization. Reject empty names, duplicate/colliding terms, invalid term types, missing parents and incompatible `ti` parents. Resolve parents after collecting all declarations; preserve main and interaction ordering separately. Copy complete graphs, not fitted row arrays.
- [ ] Add and run regressions for mutable feature inputs, helper settings, valid interaction compilation, invalid parents, numeric data refusal and explicit categorical handling. Use the current compiler/build boundary for behavioral assertions; do not merely compare repr strings.
- [ ] Run focused tests and Ruff on owned files; record commands and results, self-review and commit only owned files.

## Task 2: Bound predictors, family methods and diagnostics

**Files:** Create `src/superglm/distributional/binding.py`, `src/superglm/distributional/families/_predictors.py` and `tests/test_bound_predictors.py`. Modify all nine concrete family classes to expose helper methods through small typed mixins or direct methods.

**Interfaces:**
- Consume `TermInput` and `normalize_terms` from Task 1.
- Produce `BoundPredictor` with public `name`, `family` and defensive normalized-template access; mutation must not change a stored specification.
- Produce `bind_predictor(family, name: str, *terms: TermInput, intercept=True, link=None) -> BoundPredictor` for custom families and shared builtin binding.
- Produce `_bind_predictor_template(family, template: Predictor) -> BoundPredictor` for serialization and lower-level fixture adaptation, preserving all existing interactions and controls.
- Produce `resolve_predictors(family, predictors: Sequence[BoundPredictor]) -> tuple[DistributionalFamily, tuple[Predictor, ...]]`; validate caller identity first, snapshot family configuration, order canonical templates and validate completeness. Final link support validation may remain in existing API ownership code.
- Tweedie helper names: `mu`, `phi`, `p`. Other families expose canonical names from the spec's family table. Each method accepts `*terms`, `intercept` and `link` and returns `BoundPredictor`. Keep helper-to-canonical metadata accessible to diagnostic rendering without source inspection.

- [ ] Write the missing-helper regression and run it red:

```python
def test_tweedie_helpers_bind_actual_configured_family():
    from superglm.distributional.families.tweedie import TweedieLSS
    family = TweedieLSS(power_lower=1.08, power_upper=1.92)
    assert callable(getattr(family, "mu", None)), "family helpers are not implemented"
    assert family.mu("x").name == "mean"
    assert family.phi().name == "dispersion"
    assert family.p().family is family
```

- [ ] Implement the binder and statically declared methods. Preserve structural family protocols. Mode-dependent families expose mean/location helpers with instance validation. Custom callers may use `bind_predictor` without subclassing.
- [ ] Write and run red/green tests for completeness, arbitrary order, duplicates, foreign family instances including different Tweedie bounds, bare methods and invalid positional types. Check actual omissions and marked lines without brittle full-message snapshots. The message must contain a family-first suggested constructor, `family.p(...)` and an arrow when only power is missing, with no constant/intercept recommendation.
- [ ] Test all family parameterizations, a custom four-parameter family, model-owned family snapshots and mutable input ownership. Do not use arbitrary object repr or caller-source inspection to generate errors.
- [ ] Run `uv run pytest tests/test_bound_terms.py tests/test_bound_predictors.py -q`, focused family-contract coverage and Ruff; record results and commit owned files.

## Task 3: Public constructor, persistence and call-site migration

**Files:** Modify `src/superglm/distributional/api.py`, root and distributional exports, existing `SuperLSS` constructor call sites under `tests/` and `benchmarks/`. Create `tests/test_bound_superlss.py` and, if needed, a clearly named test-only fixture adapter under `tests/`.

**Interfaces:**
- Consume `resolve_predictors` and `_bind_predictor_template` from Task 2.
- Produce public `SuperLSS(family, /, *predictors: BoundPredictor, weight_semantics="prior", discrete=False, n_bins=256, separation="warn", coefficient_curvature="observed")`.
- Export bound term helpers, `BoundPredictor` and `bind_predictor` through `superglm`. Replace lazy family imports at the root with statically visible imports if needed for consumer typing. Preserve documented lower-level numerical contracts.

- [ ] Write and run a failing constructor test using actual helper calls:

```python
def test_family_first_constructor_orders_named_predictors():
    from superglm.distributional.families.tweedie import TweedieLSS
    from superglm import SuperLSS
    family = TweedieLSS(power_lower=1.08, power_upper=1.92)
    model = SuperLSS(family, family.p(), family.mu("x"), family.phi())
    assert tuple(p.name for p in model.predictors) == ("mean", "dispersion", "power")
    assert model.family.to_config() == family.to_config()
```

- [ ] Integrate construction with current validation and owned predictors. Preserve all option meanings. Refuse old `predictors=` construction with an actionable migration message; do not silently accept old dict/tuple public input.
- [ ] Update deserialization using `_bind_predictor_template` and the new constructor, preserving artifact schema and certified model state.
- [ ] Migrate public boundary tests to the actual new interface. Numerical tests whose inputs deliberately exercise internal `Predictor` templates may use a test-only adapter that binds those templates then calls the real new constructor. Do not add a production compatibility method solely for tests. Keep unrelated expected numerical outputs and tolerances unchanged. Update benchmarks to executable new construction without test imports.
- [ ] Add real fixed-smoothing and REML parity tests with internal-baseline fixtures, including discrete execution, interactions, predictions, covariance and round trips. Include old trusted artifact loading where existing fixtures support it. Reject wrong input with specific errors at the public boundary.
- [ ] Run focused public API, prediction, discrete, serialization and new binding suites. Review all remaining public `SuperLSS(` call sites for stale syntax. Record results and commit owned files.

## Task 4: User docs, installed typing and complete verification

**Files:** Modify `README.md`, `docs/models/distributional.md`, `docs/models/distributional-inference.md`, `docs/distributional-family-development.md`; create `src/superglm/py.typed`, focused consumer typing fixtures and a packaging check under tests/scripts as appropriate. Do not rewrite historical research receipts.

**Interfaces:** Consume the completed public constructor and family helpers. Document canonical parameter names used by prediction, offsets and results separately from Tweedie's short construction helpers.

- [ ] Rewrite primary examples around the complete family-first declaration. Show missing-predictor arrows with `...`, explicit empty helpers, custom binding and supported `ti` semantics. Keep the single model entry point and numeric string meaning clear.
- [ ] Add the packaged marker, build a wheel and inspect its contents. Use an isolated installed-wheel consumer check with a pinned available type checker: valid Tweedie and Gaussian construction must pass; misspelled helpers and invalid term input must fail. Validate statically visible imports, avoiding an `Any` fallback that would make negative checks meaningless.
- [ ] Run focused new tests and public regression suites, then applicable complete repository checks: `uv run pytest tests/ -q -m "not slow"`, Ruff check/format, `uv lock --check`, `uv pip check`, and `uv run python run_test.py`. Use `SUPERGLM_REQUIRE_DATA=1` when the three mandatory real-data suites are included; obtain the local dataset using the documented fetch command if needed. Run the full suite as practical, reporting legitimate optional skips explicitly rather than calling them executed coverage.
- [ ] Review the complete diff for numerical-path changes, stale public examples, version changes and generated artifacts. Record exact evidence and any environmental blockers, then commit the docs/typing changes and necessary fixes.

## Review and delivery

Each implementation task receives an independent task review. A final review checks the whole branch, including migrated tests, ownership boundaries, typing negatives and serialization. Address material findings with focused regressions. Keep the branch and worktree available for user review; do not merge or publish without an explicit instruction.
