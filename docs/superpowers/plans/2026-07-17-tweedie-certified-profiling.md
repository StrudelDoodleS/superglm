# Certified Tweedie Profiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure only exact, converged fixed-power records can determine a Tweedie estimate and restrict likelihood-ratio inference to statistically valid profiles.

**Architecture:** Extend immutable profile records with explicit certification reasons and classify each result by inference kind. Search methods keep rejected evaluations for diagnostics but select only certified records, evaluate configured endpoints consistently, and carry effective bounds into CI consumers. Typed profile failures retain trace evidence without mutating caller state.

**Tech Stack:** Python dataclasses and `Literal`, NumPy, pandas, SciPy optimizers, pytest.

---

## File map

- Modify `src/superglm/profiling/tweedie.py`: inference kinds, record certification, typed errors, search endpoint behavior, effective bounds, and CI eligibility.
- Modify `src/superglm/profiling/__init__.py`: public profile error/result type exports.
- Modify `src/superglm/__init__.py`: stable facade exports for the typed error and inference kind.
- Modify `src/superglm/profiling/_reporting.py`: inference-kind-aware cached CI status and method labels.
- Create `tests/test_tweedie_profile_certification.py`: focused certification, search, and inference regressions.
- Modify `tests/test_tweedie_profile.py`: update legacy selection/finalization contracts.
- Modify `tests/test_profile_ci.py`: CI eligibility and effective-range coverage.

### Task 1: Define inference kinds and typed profile failure

**Files:**
- Modify: `src/superglm/profiling/tweedie.py:2501-2728`
- Modify: `src/superglm/profiling/__init__.py`
- Modify: `src/superglm/__init__.py`
- Create: `tests/test_tweedie_profile_certification.py`

- [ ] **Step 1: Write failing public-type and classification tests**

```python
import pytest

from superglm import TweedieProfileError, TweedieProfileRecord
from superglm.profiling.tweedie import _resolve_inference_kind


@pytest.mark.parametrize(
    ("fit_mode", "phi_method", "has_penalty", "has_constraints", "expected"),
    [
        ("fit", "mle", False, False, "exact_mle"),
        ("fit", "mle", False, True, "constrained_profile"),
        ("fit", "pearson", False, False, "pearson_plugin"),
        ("fit", "mle", True, False, "penalized_plugin"),
        ("fit_reml", "mle", True, False, "reml_plugin"),
    ],
)
def test_resolve_inference_kind(
    fit_mode, phi_method, has_penalty, has_constraints, expected
):
    assert (
        _resolve_inference_kind(
            fit_mode=fit_mode,
            phi_method=phi_method,
            has_penalty=has_penalty,
            has_constraints=has_constraints,
            density_exact=True,
        )
        == expected
    )


def test_profile_error_preserves_trace_and_rejections():
    trace = (
        TweedieProfileRecord(
            step=0,
            p=1.1,
            phi=1.0,
            nll=2.0,
            source="grid",
            fit_converged=True,
            phi_converged=True,
            density_exact=False,
            density_certified=False,
            selectable=False,
            rejection_reasons=("density_not_exact", "density_not_certified"),
        ),
    )
    rejection = {1.1: ("density_not_exact", "density_not_certified")}
    error = TweedieProfileError("no certified record", trace, rejection)
    assert error.search_trace == trace
    assert error.rejections == rejection
```

- [ ] **Step 2: Verify imports fail**

Run: `rtk pytest tests/test_tweedie_profile_certification.py -k 'inference_kind or profile_error'`

Expected: import fails because `TweedieProfileError` and `_resolve_inference_kind` do not exist.

- [ ] **Step 3: Add stable types and resolver**

```python
TweedieInferenceKind = Literal[
    "exact_mle",
    "constrained_profile",
    "pearson_plugin",
    "penalized_plugin",
    "reml_plugin",
    "approximate",
]

TweedieProfileRejectionReason = Literal[
    "p_not_finite",
    "objective_not_finite",
    "phi_not_positive_finite",
    "fit_not_converged",
    "phi_not_converged",
    "density_not_exact",
    "density_not_certified",
]


@dataclass(frozen=True)
class TweedieProfileRecord:
    step: int
    p: float
    phi: float
    nll: float
    source: str
    fit_converged: bool
    phi_converged: bool
    density_exact: bool
    density_certified: bool
    selectable: bool
    rejection_reasons: tuple[TweedieProfileRejectionReason, ...]


class TweedieProfileError(RuntimeError):
    def __init__(self, message, search_trace=(), rejections=None):
        super().__init__(message)
        self.search_trace = tuple(search_trace)
        self.rejections = {
            float(p): tuple(str(reason) for reason in reasons)
            for p, reasons in (rejections or {}).items()
        }


def _resolve_inference_kind(
    *, fit_mode, phi_method, has_penalty, has_constraints, density_exact
):
    if not density_exact:
        return "approximate"
    if fit_mode == "fit_reml":
        return "reml_plugin"
    if has_penalty:
        return "penalized_plugin"
    if has_constraints:
        return "constrained_profile"
    if phi_method == "pearson":
        return "pearson_plugin"
    return "exact_mle"
```

Add `records: tuple[TweedieProfileRecord, ...]`, `inference_kind: TweedieInferenceKind`,
`searched_bounds: tuple[float, float]`, and `density_certified: bool` to
`TweedieProfileResult`. Add a read-only `supports_likelihood_ratio` property requiring an interior,
converged, certified `exact_mle` with no power or dispersion boundary. Derive conservative
compatibility defaults in `__setstate__`; a legacy result is never presumed LR-capable from
`phi_method="mle"` alone. Export the error, record, rejection-reason, and inference-kind types
through both package facades. `records` is authoritative; `search_trace` remains its mutable
backward-compatible DataFrame projection.

```python
@property
def supports_likelihood_ratio(self):
    return bool(
        self.inference_kind == "exact_mle"
        and self.converged
        and self.fit_converged
        and self.phi_converged
        and self.density_exact
        and self.density_certified
        and self.outer_boundary is None
        and not self.phi_boundary
    )
```

- [ ] **Step 4: Run and commit**

Run: `rtk pytest tests/test_tweedie_profile_certification.py -k 'inference_kind or profile_error'`

Expected: all selected tests pass.

```bash
rtk git add src/superglm/profiling/tweedie.py src/superglm/profiling/__init__.py src/superglm/__init__.py tests/test_tweedie_profile_certification.py
rtk git commit -m "Classify Tweedie profile inference"
```

### Task 2: Make record certification strict and explain every rejection

**Files:**
- Modify: `src/superglm/profiling/tweedie.py:3128-3223,3726-3895`
- Modify: `tests/test_tweedie_profile_certification.py`
- Modify: `tests/test_tweedie_profile.py`

- [ ] **Step 1: Add failing rejection tests**

Build records with the existing test helpers and assert each invalid dimension is rejected:

```python
@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"fit_converged": False}, "fit_not_converged"),
        ({"phi_converged": False}, "phi_not_converged"),
        ({"objective_finite": False}, "objective_not_finite"),
        ({"density_exact": False}, "density_not_exact"),
        ({"density_certified": False}, "density_not_certified"),
        ({"phi": -1.0}, "phi_not_positive_finite"),
    ],
)
def test_profile_record_rejection_reasons(record_factory, change, reason):
    record = record_factory(**change)
    assert reason in _profile_record_rejection_reasons(record)
    assert not _profile_record_is_selectable(record)


def test_lower_nll_nonconverged_record_cannot_beat_certified_record(profile_context_factory):
    ctx = profile_context_factory(
        records=[
            {"p": 1.4, "nll": 0.1, "phi_converged": True, "density_exact": True},
            {"p": 1.5, "nll": 0.0, "phi_converged": False, "density_exact": True},
        ]
    )
    winner = _best_finite_profile_record(ctx, method="grid", searched_bounds=(1.4, 1.5))
    assert winner.p == 1.4
```

- [ ] **Step 2: Demonstrate current finite-only selection fails**

Run: `rtk pytest tests/test_tweedie_profile_certification.py -k 'rejection or cannot_beat'`

Expected: the nonconverged `p=1.5` record wins or is reported selectable.

- [ ] **Step 3: Implement machine-readable reasons**

Add exactness fields directly to `_ProfileEvaluation` at creation time so certification does not
reconstruct mutable diagnostics later:

```python
def _profile_record_rejection_reasons(record):
    reasons = []
    if not np.isfinite(record.p):
        reasons.append("p_not_finite")
    if not np.isfinite(record.nll) or not record.phi_result.objective_finite:
        reasons.append("objective_not_finite")
    if not np.isfinite(record.phi) or record.phi <= 0.0:
        reasons.append("phi_not_positive_finite")
    if not record.fit_converged:
        reasons.append("fit_not_converged")
    if not record.phi_result.converged:
        reasons.append("phi_not_converged")
    if not record.density_exact:
        reasons.append("density_not_exact")
    if not record.density_certified:
        reasons.append("density_not_certified")
    return tuple(reasons)


def _profile_record_is_selectable(record):
    return not _profile_record_rejection_reasons(record)
```

Materialize one frozen scalar `TweedieProfileRecord` from every internal `_ProfileEvaluation`; never
copy row-scale `mu` into it. Add `selectable`, `rejection_reasons`, and `density_certified` columns to
the compatibility `_materialize_profile_trace_row` projection.

When no selectable record exists, `_best_finite_profile_record` raises `TweedieProfileError` with
the materialized trace and `{p: reasons}` mapping. Update its error message to report counts by
reason rather than claiming every finite record is valid.

- [ ] **Step 4: Run neighboring profile tests and commit**

Run: `rtk pytest tests/test_tweedie_profile_certification.py tests/test_tweedie_profile.py -k 'selectable or invalid or finalize or convergence or density'`

Expected: strict selection and updated legacy contracts pass.

```bash
rtk git add src/superglm/profiling/tweedie.py tests/test_tweedie_profile_certification.py tests/test_tweedie_profile.py
rtk git commit -m "Reject uncertified Tweedie profiles"
```

### Task 3: Carry effective penalty and constraint semantics into results

**Files:**
- Modify: `src/superglm/profiling/tweedie.py:3237-3724,3780-3871,4174-4337`
- Modify: `tests/test_tweedie_profile_certification.py`

- [ ] **Step 1: Add failing real-model inference-kind tests**

```python
@pytest.mark.parametrize(
    ("model_kwargs", "fit_mode", "phi_method", "expected"),
    [
        ({"selection_penalty": 0.0}, "fit", "mle", "exact_mle"),
        ({"selection_penalty": 1000.0}, "fit", "mle", "penalized_plugin"),
        ({"selection_penalty": 0.0}, "fit", "pearson", "pearson_plugin"),
        ({"selection_penalty": 0.0}, "fit_reml", "mle", "reml_plugin"),
    ],
)
def test_real_profile_reports_inference_kind(
    simple_tweedie_data, model_kwargs, fit_mode, phi_method, expected
):
    X, y = simple_tweedie_data
    model = SuperGLM(family=Tweedie(1.5), features={"x": Numeric()}, **model_kwargs)
    result = estimate_tweedie_p(
        model, X, y, fit_mode=fit_mode, phi_method=phi_method, method="grid", grid=[1.4, 1.5]
    )
    assert result.inference_kind == expected
```

- [ ] **Step 2: Verify current results have no inference kind**

Run: `rtk pytest tests/test_tweedie_profile_certification.py -k real_profile_reports`

Expected: `AttributeError: TweedieProfileResult has no attribute inference_kind`.

- [ ] **Step 3: Compute effective regularization from fitted groups**

At context construction, calculate `has_penalty` from actual selected/penalized groups and nonzero
selection/ridge/smoothing parameters, not merely the presence of a default `lambda2` value. Calculate
`has_constraints` from active feature/group constraint matrices. Store both booleans and `fit_mode`
on the context. Resolve the result kind in `_finalize_profile_record` from those fields plus
`phi_method` and winning-record exactness.

Inference precedence is: approximate density, REML, active penalty, active constraint, Pearson
dispersion, then exact MLE. This makes the primary label describe the strongest departure from a
regular likelihood while `phi_method` and constraint fields retain the other traits.

- [ ] **Step 4: Run and commit**

Run: `rtk pytest tests/test_tweedie_profile_certification.py -k inference_kind`

Expected: all synthetic and real-model classification tests pass.

```bash
rtk git add src/superglm/profiling/tweedie.py tests/test_tweedie_profile_certification.py
rtk git commit -m "Report Tweedie profile semantics"
```

### Task 4: Evaluate endpoints consistently and store search support

**Files:**
- Modify: `src/superglm/profiling/tweedie.py:3762-3769,3965-4171`
- Modify: `tests/test_tweedie_profile_certification.py`

- [ ] **Step 1: Add transformed-optimizer endpoint regression**

```python
def test_profile_optimizer_evaluates_and_selects_exact_endpoint(monotone_profile_context):
    result = _search_profile_opt(
        monotone_profile_context,
        p_bounds=(1.1, 1.9),
        optimizer="L-BFGS-B",
        xatol=1e-6,
        maxiter=50,
    )
    assert 1.1 in monotone_profile_context._evaluation_cache
    assert 1.9 in monotone_profile_context._evaluation_cache
    assert result.p_hat == 1.1
    assert result.outer_boundary == "lower"
    assert result.searched_bounds == (1.1, 1.9)
```

- [ ] **Step 2: Verify the current optimizer omits endpoints**

Run: `rtk pytest tests/test_tweedie_profile_certification.py -k exact_endpoint`

Expected: neither exact endpoint is cached and `outer_boundary` is `None`.

- [ ] **Step 3: Add common endpoint probing and tolerance-aware classification**

Create `_evaluate_search_endpoints(ctx, bounds, source)` and call it before every search strategy,
including `profile_opt`. Grid with an explicit array uses `(min(grid), max(grid))` as its effective
bounds and evaluates those exact entries. Use:

```python
def _outer_boundary_label(p, bounds, xatol):
    lo, hi = bounds
    atol = max(2.0 * float(xatol), 64.0 * np.finfo(np.float64).eps * max(1.0, abs(lo), abs(hi)))
    if np.isclose(p, lo, rtol=0.0, atol=atol):
        return "lower"
    if np.isclose(p, hi, rtol=0.0, atol=atol):
        return "upper"
    return None
```

Pass `xatol` through finalization. Populate both `searched_bounds` and `_ci_p_range` from the same
validated tuple.

- [ ] **Step 4: Run all search-method tests and commit**

Run: `rtk pytest tests/test_tweedie_profile_certification.py tests/test_tweedie_profile.py -k 'brent or grid or profile_opt or boundary or endpoint'`

Expected: all methods evaluate effective endpoints and report boundaries consistently.

```bash
rtk git add src/superglm/profiling/tweedie.py tests/test_tweedie_profile_certification.py tests/test_tweedie_profile.py
rtk git commit -m "Certify Tweedie search boundaries"
```

### Task 5: Restrict likelihood-ratio confidence intervals

**Files:**
- Modify: `src/superglm/profiling/tweedie.py:2729-2846,4357-4978`
- Modify: `src/superglm/profiling/_reporting.py`
- Modify: `tests/test_profile_ci.py`
- Modify: `tests/test_tweedie_profile_certification.py`

- [ ] **Step 1: Add failing CI eligibility tests**

```python
@pytest.mark.parametrize(
    ("inference_kind", "boundary", "match"),
    [
        ("penalized_plugin", None, "penalized plug-in"),
        ("reml_plugin", None, "REML plug-in"),
        ("pearson_plugin", None, "Pearson plug-in"),
        ("constrained_profile", None, "constrained profile"),
        ("approximate", None, "approximate profile"),
        ("exact_mle", "lower", "boundary profile"),
    ],
)
def test_ci_rejects_nonregular_profile(profile_result_factory, inference_kind, boundary, match):
    result = profile_result_factory(
        inference_kind=inference_kind,
        outer_boundary=boundary,
        searched_bounds=(1.01, 1.99),
    )
    with pytest.raises(RuntimeError, match=match):
        result.ci()


def test_ci_range_comes_from_effective_search_bounds(profile_result_factory):
    result = profile_result_factory(
        p_hat=1.99,
        searched_bounds=(1.98, 1.999),
        inference_kind="exact_mle",
    )
    assert result._ci_p_range == (1.98, 1.999)
```

- [ ] **Step 2: Demonstrate penalized and near-boundary failures**

Run: `rtk pytest tests/test_profile_ci.py tests/test_tweedie_profile_certification.py -k 'nonregular or effective_search_bounds'`

Expected: penalized/REML/constrained results enter ordinary LR code and `p_hat=1.99` conflicts with
the hard-coded `(1.02, 1.98)` range.

- [ ] **Step 3: Centralize the CI gate**

Add `_validate_lr_inference` and call it before any cache lookup or objective evaluation:

```python
def _validate_lr_inference(result):
    labels = {
        "penalized_plugin": "penalized plug-in",
        "reml_plugin": "REML plug-in",
        "pearson_plugin": "Pearson plug-in",
        "constrained_profile": "constrained profile",
        "approximate": "approximate profile",
    }
    if result.inference_kind != "exact_mle":
        raise RuntimeError(
            f"Likelihood-ratio CI is unavailable for {labels[result.inference_kind]}; "
            "use calibrated bootstrap inference."
        )
    if result.outer_boundary is not None or result.phi_boundary:
        raise RuntimeError("Likelihood-ratio CI is unavailable for a boundary profile")
    if not result.supports_likelihood_ratio:
        raise RuntimeError("Likelihood-ratio CI requires a converged certified exact profile")
```

Apply the same exact-record certification to every CI probe. A rejected CI probe produces the
existing typed CI evaluation error and cannot form an endpoint.

Update cached-report status literals to cover each unavailable kind without evaluating the
objective.

- [ ] **Step 4: Run CI and reporting tests and commit**

Run: `rtk pytest tests/test_profile_ci.py tests/test_tweedie_profile_certification.py tests/test_tweedie_profile.py -k 'ci or inference or boundary'`

Expected: LR inference works only for certified interior `exact_mle` results.

```bash
rtk git add src/superglm/profiling/tweedie.py src/superglm/profiling/_reporting.py tests/test_profile_ci.py tests/test_tweedie_profile_certification.py tests/test_tweedie_profile.py
rtk git commit -m "Restrict Tweedie likelihood inference"
```

### Task 6: Run the certified-profile workstream gate

**Files:**
- Modify: `tests/test_tweedie_profile_certification.py`

- [ ] **Step 1: Add a public seed-101 nonselection assertion**

Extend the numerical workstream regression to assert every rejected trace row has nonempty
`rejection_reasons`, the old `p=1.05` approximate row is not selectable, and the selected interior
row is exact and converged.

- [ ] **Step 2: Run focused tests**

Run: `rtk pytest tests/test_tweedie_profile_certification.py tests/test_tweedie_profile.py tests/test_profile_ci.py tests/test_weighted_forwarding.py`

Expected: all focused profile, CI, and forwarding tests pass.

- [ ] **Step 3: Run style and typing checks**

Run: `rtk ruff check src/superglm/profiling/tweedie.py src/superglm/profiling/_reporting.py src/superglm/profiling/__init__.py src/superglm/__init__.py tests/test_tweedie_profile_certification.py tests/test_tweedie_profile.py tests/test_profile_ci.py`

Run: `rtk proxy uv run mypy --follow-imports=skip src/superglm/profiling/tweedie.py src/superglm/profiling/_reporting.py`

Expected: Ruff passes and focused typing has no errors introduced by this workstream.

- [ ] **Step 4: Commit gate-only test refinements**

```bash
rtk git add tests/test_tweedie_profile_certification.py
rtk git commit -m "Cover certified Tweedie profile traces"
```
