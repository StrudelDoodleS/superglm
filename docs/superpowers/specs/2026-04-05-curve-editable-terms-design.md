# Curve-Editable Terms: Design Spec

## Goal

Make curve-like terms inspectable and locally editable in SuperGLM + ModelForge. The encouraged
workflow is:

1. Fit a baseline spline with ordinary `Spline(...)` and REML.
2. Inspect the fitted curve.
3. Paint local repairs: mark regions as flat, linear, or free.
4. Refit — edited regions obey the constraint, unedited regions stay close to baseline.

This is a **repair/refit workflow on top of a baseline spline**, not a standalone segmented
spline family. The editing capability lives in ModelForge (private). SuperGLM provides the
engine hooks (LambdaPolicy, CurveState).

## Non-Goals

- No hard shape constraints in v0
- No frontend or UI work in this spec
- SegmentedSpline is NOT part of public SuperGLM — it is private to ModelForge

## What Goes Where

### Public SuperGLM

- **LambdaPolicy** — per-component lambda control in REML (PR 1)
- **CurveState** — structured export of fitted smooth term state (PR 2)
- Nothing editor-specific. These are general engine capabilities.

### Private ModelForge (`backend/app/superglm_ext/`)

- **SegmentedSpline** — B-spline with joins, regions, baseline anchoring
- **Region** — interval-local behavior spec (free / flat / linear)
- **Join** — breakpoint continuity control
- Builder layer, config wiring, persistence, export

---

## Architecture: Two SuperGLM PRs + Private ModelForge Layer

### PR 1: LambdaPolicy (SuperGLM)

General REML mechanism for per-penalty-component control. Independently useful for all existing
multi-penalty terms (`m=(1,2)`, `select=True`, tensor products).

### PR 2: CurveState (SuperGLM)

Structured export of fitted smooth term state. Works on all smooth terms: `Spline`,
`OrderedCategorical(basis="spline")`, and later SegmentedSpline.

### Private: SegmentedSpline (ModelForge)

B-spline with joins, regions, and baseline-preserving refit. Implements SuperGLM's FeatureSpec
protocol. Cubic-only in v0.

**Key capability not yet implemented: baseline anchoring.** When a user edits only [30,45] to
be flat, the free regions should stay close to the original baseline curve. Without this, the
optimizer distorts free segments to accommodate constrained ones.

Baseline anchoring is implemented as a penalty: penalise `||f(x) - f0(x)||^2` on a grid outside
the edited intervals, where `f0` is the baseline fitted curve. This is baked into the total
penalty matrix alongside the region derivative penalties.

**Knot assembly policy:** Auto knots near breakpoints are cleared (half-median-spacing exclusion
radius) before inserting repeated knots. This prevents knot clustering that gives the optimizer
excessive local flexibility near boundaries. Matches the JSX prototype behavior.

**Region penalties:** Interval-local derivative penalties via Gauss-Legendre quadrature:
- Flat: `int_lo^hi B'(x)^T B'(x) dx` (penalise first derivative)
- Linear: `int_lo^hi B''(x)^T B''(x) dx` (penalise second derivative)

These are baked into the total penalty matrix with fixed weight, not exposed as REML-optimizable
components. Will migrate to `LambdaPolicy.fixed()` when that lands upstream.

---

## PR 1: LambdaPolicy

### Type

```python
@dataclass(frozen=True)
class LambdaPolicy:
    mode: Literal["estimate", "fixed"]
    value: float | None = None  # required when mode="fixed"

    @classmethod
    def estimate(cls) -> LambdaPolicy:
        return cls(mode="estimate")

    @classmethod
    def fixed(cls, value: float) -> LambdaPolicy:
        return cls(mode="fixed", value=value)

    @classmethod
    def off(cls) -> LambdaPolicy:
        return cls.fixed(0.0)
```

Two modes only. `off()` is syntactic sugar for `fixed(0.0)`.

### Feature-Level API

All smooth/spline-like feature constructors gain `lambda_policy=` (Spline, OrderedCategorical,
SegmentedSpline — not Numeric or plain Categorical):

```python
# Broadcast to all components
Spline(kind="bs", k=10, lambda_policy=LambdaPolicy.fixed(0.5))

# Per-component
Spline(kind="bs", k=10, m=(1, 2), lambda_policy={
    "d1": LambdaPolicy.estimate(),
    "d2": LambdaPolicy.fixed(1.0),
})
```

Semantics:
- Single `LambdaPolicy` broadcasts to all penalty components on the term.
- Dict keyed by component name overrides per-component. Unspecified components default to
  `estimate`.
- Unknown component keys raise `ValueError`.
- Component names (`d1`, `d2`, `null`, `wiggle`, etc.) become stable public API once dict
  policies exist. Must be documented.

### Fit Behavior

| Call | `estimate` | `fixed(v)` | `off()` |
|------|------------|------------|---------|
| `fit()` | Uses `model.lambda2` | Uses `v` | Uses `0.0` |
| `fit_reml()` | Optimized by REML (init from `lambda2_init`) | Uses `v` | Uses `0.0` |

### Data Flow

1. Feature's `build()` attaches lambda policy metadata to `GroupInfo` (new field).
2. Policy metadata is carried through design matrix / group-matrix construction.
3. Fit plumbing in `fit_ops.py` reads policies and assembles a full numeric `lambdas: dict[str, float]`
   for all components (estimated + fixed).
4. REML receives the full lambdas dict but only optimizes the estimated subset. Fixed components
   stay at their value throughout.
5. `_build_penalty_matrix()` remains purely numeric. It already skips zero lambdas, so `off()`
   works naturally.
6. The full REML objective, gradient, Hessian, and log-det computations use **all** penalty
   components (estimated + fixed). Fixed components contribute to `S(lambda)`, `log|S|_+`,
   `log|H|`, and all derivative terms — they are part of the penalized system. Only the
   **optimizer state vector** is reduced: Newton steps, Fellner-Schall updates, and convergence
   checks operate over the estimated subset only. This distinction matters when multiple penalty
   components share a coefficient block — dropping fixed components from the algebra would give
   wrong derivatives for the estimated components.

### Files Changed

- `src/superglm/types.py` — add `LambdaPolicy` dataclass, add lambda policy field to `GroupInfo`
- `src/superglm/features/spline.py` — accept `lambda_policy=` on all spline constructors, attach
  to `GroupInfo`
- `src/superglm/model/fit_ops.py` — read policies, partition components, initialize lambdas
- `src/superglm/reml/direct.py` — reduce Newton state vector to estimated components only;
  full penalty algebra unchanged
- `src/superglm/reml/runner.py` — Fellner-Schall updates only for estimated components
- `src/superglm/reml/efs.py` — EFS updates only for estimated components
- `src/superglm/reml/discrete.py` — POI updates only for estimated components
- `src/superglm/reml/gradient.py` — gradient/Hessian computed over all components; optimizer
  extracts the estimated subset for step computation
- `src/superglm/reml/penalty_algebra.py` — log-det computed over all components (no change
  to algebra, only to which lambdas the optimizer is allowed to update)

### Tests

- Spline with `lambda_policy=LambdaPolicy.fixed(1.0)`: verify lambda is exactly 1.0 after fit_reml
- Spline with `m=(1,2)`, mixed policy: d1 estimated, d2 fixed — verify d2 unchanged, d1 optimized
- Spline with `lambda_policy=LambdaPolicy.off()`: verify zero penalty, term is unpenalized
- Regression: existing `Spline()` without lambda_policy unchanged
- All REML paths (direct, EFS, runner, discrete) tested with mixed policies
- `fit()` (non-REML) respects fixed values

---

## PR 2: CurveState

### Types

New module: `src/superglm/curve_state.py`

```python
@dataclass(frozen=True)
class PenaltyInfo:
    name: str                       # "d1", "d2", "null", etc.
    penalty_semantics: str          # "difference", "integrated_derivative", ...
    component_role: str | None      # "selection", "wiggle", etc.
    lambda_final: float             # final smoothing parameter value
    lambda_policy: LambdaPolicy     # policy that was used
    rank: float                     # penalty rank


@dataclass(frozen=True)
class JoinInfo:
    at: float                       # breakpoint location
    continuity: str                 # "c0", "c1", "c2", "jump"
    knot_multiplicity: int          # resolved multiplicity in the knot vector


@dataclass(frozen=True)
class CurveState:
    # Identity
    name: str
    feature_type: str               # "spline", "ordered_categorical", "segmented_spline"

    # Basis geometry (nullable — not all smooth terms have all fields)
    basis_kind: str | None          # "bs", "cr", "ns", None
    domain: tuple[float, float] | None
    knots: NDArray | None
    degree: int | None
    n_basis_raw: int | None         # pre-constraint basis size
    n_coefficients_fit: int         # post-projection coefficient count

    # Fitted values
    coefficients_raw_basis: NDArray | None  # in original (non-reparametrised) basis
    projection: NDArray | None              # constraint/identifiability projection

    # Penalty structure
    penalties: list[PenaltyInfo]
    edf: float | None

    # Evaluation
    eval_grid: NDArray              # default evaluation grid
    eval_values: NDArray            # fitted curve on grid
    value_scale: str = "linear_predictor"   # generic, not link-specific

    # Type-specific
    category_map: dict[str, float] | None = None   # for OrderedCategorical
    joins: list[JoinInfo] | None = None             # for SegmentedSpline

    def to_dict(self) -> dict: ...

    @classmethod
    def from_dict(cls, d: dict) -> CurveState: ...
```

### Model API

```python
model.curve_state(name: str, n_points: int = 200) -> CurveState
```

- Raises `TypeError` if the term is not a smooth term (e.g. `Numeric`, `Categorical`).
- Delegates to the feature spec internally for type-specific state extraction.
- Pulls penalty info from `model._reml_lambdas` and design matrix metadata.
- Pulls edf from existing `state_ops` machinery.

### Feature Hook

Each smooth feature type implements an internal method for state extraction:

```python
def _export_curve_state(self, beta, lambdas, edf, n_points) -> CurveState
```

The model calls this, passing the relevant coefficient slice and metadata. The feature knows its
own basis geometry and how to reconstruct.

### Files Changed

- New: `src/superglm/curve_state.py` — `CurveState`, `PenaltyInfo` dataclasses + JSON helpers
- `src/superglm/model/api.py` — add public `curve_state()` method (helper logic in base.py or
  a dedicated module)
- `src/superglm/features/spline.py` — add `_export_curve_state()` to `_SplineBase`
- `src/superglm/features/ordered_categorical.py` — add `_export_curve_state()`

### Tests

- `model.curve_state("age")` on a fitted Spline: verify all fields populated correctly
- `model.curve_state("region")` on OrderedCategorical(basis="spline"): verify category_map present
- JSON round-trip: `CurveState.from_dict(cs.to_dict())` matches original
- Raises `TypeError` on non-smooth terms
- Verify `n_basis_raw` vs `n_coefficients_fit` are correct for constrained splines

---

## Private: SegmentedSpline (ModelForge)

Lives in `model-forge/backend/app/superglm_ext/`. NOT in public SuperGLM.

### Editing Workflow

```
1. User fits baseline:   model.fit_reml(X, y)  # ordinary Spline
2. User inspects curve:  model.curve_state("age")
3. User paints edits:    regions=[Region(30,45,"flat"), Region(45,60,"linear")]
4. User refits:          new_model.fit_reml(X, y)
   - edited regions obey flat/linear constraints
   - free regions anchor toward the baseline curve f0(x)
```

### Constructor

```python
class SegmentedSpline:
    def __init__(
        self,
        n_knots: int = 10,
        knot_strategy: str = "quantile",
        knots: ArrayLike | None = None,
        joins: list[Join] | None = None,
        regions: list[Region] | None = None,
        baseline_curve: tuple[NDArray, NDArray] | None = None,  # (x_grid, f0_values)
        m: int | tuple[int, ...] = 2,
        penalty: str = "ssp",
        boundary: tuple[float, float] | None = None,
        extrapolation: str = "clip",
    ): ...
```

Cubic-only in v0. Internal `degree = 3`.

### Join

```python
@dataclass(frozen=True)
class Join:
    at: float
    continuity: str = "c0"  # "c0", "c1", "c2", "jump" (experimental)
```

Continuity → knot multiplicity (cubic):
- `"c2"` → 1 (normal knot)
- `"c1"` → 2 (smooth 1st deriv, kink in 2nd)
- `"c0"` → 3 (continuous, visible kink)
- `"jump"` → 4 (discontinuous, experimental)

### Region

```python
@dataclass(frozen=True)
class Region:
    lo: float
    hi: float
    behavior: str = "free"  # "free", "flat", "linear"
```

### Knot Assembly (three-phase, matches JSX prototype)

1. Compute base interior knots (uniform/quantile).
2. Collect breakpoint specs: explicit `Join.at` wins over region-boundary defaults.
   Region boundaries default to C0 if no explicit join exists there.
3. Clear auto knots near breakpoints (half-median-spacing exclusion radius), then insert
   repeated knots. This prevents knot clustering that gives the optimizer excessive local
   freedom near boundaries.

### Penalty Structure

**Global:** P-spline coefficient-difference penalty `D_m^T D_m`.

**Region-local:** Interval-local derivative penalties via Gauss-Legendre quadrature:
- Flat: `int_lo^hi B'(x)^T B'(x) dx`
- Linear: `int_lo^hi B''(x)^T B''(x) dx`

**Baseline anchor (NOT YET IMPLEMENTED — next milestone):**
Outside edited regions, penalise `||f(x) - f0(x)||^2` on a grid. This is the critical piece
that prevents free-region distortion. Without it, the optimizer is free to contort free segments
to accommodate constrained ones.

Implementation: evaluate the baseline curve `f0` and the new basis `B` on a grid of points
outside all region intervals. Build a penalty `(B_out @ beta - f0_out)^T (B_out @ beta - f0_out)`
which expands to a quadratic in beta: `beta^T (B_out^T B_out) beta - 2 f0_out^T B_out beta + const`.
The quadratic term `B_out^T B_out` adds to the penalty matrix. The linear term shifts the
penalty minimum away from zero.

Region and anchor penalties are baked into the total penalty matrix with fixed weights, not
exposed as REML-optimizable components. Will migrate to `LambdaPolicy.fixed()` when available.

### select=True (Experimental)

Accepted but experimental. Null-space decomposition changes with repeated knots.
`select=True` operates on penalty structure, not join geometry — it cannot "turn off" a breakpoint.

### Code Reuse

Self-contained in ModelForge. Duplicates small knot-placement and penalty code rather than
depending on `_SplineBase` internals. Imports SSP and DiscretizedSSPGroupMatrix from SuperGLM.

### Current Implementation Status

**Done:**
- Join knot assembly with multiplicity control
- Region-local derivative penalties (quadrature-based)
- Knot clearing around breakpoints
- FeatureSpec protocol (build/transform/reconstruct)
- Builder layer, TermConfig/TermPayload wiring, persistence, export
- Numerical regression tests (flat slope, linear curvature, knot clearing)

**Not done:**
- Baseline anchoring (next milestone)
- `select=True` validation with repeated knots
- `discrete=True` path
- CurveState export integration

---

## Verification Plan

### PR 1 (LambdaPolicy)

1. Run full test suite: `pytest`
2. Manual test: fit a Spline with `m=(1,2)`, `lambda_policy={"d1": LambdaPolicy.fixed(0.5)}`.
   Verify d1 lambda is exactly 0.5 after REML. Verify d2 was optimized.
3. Benchmark: fit MTPL2 (678k rows) with and without lambda_policy. No regression.

### PR 2 (CurveState)

1. Run full test suite: `pytest`
2. Manual test: fit a model, call `model.curve_state("age")`. Inspect all fields.
3. Round-trip: `CurveState.from_dict(cs.to_dict())` — verify equality.

### SegmentedSpline (ModelForge)

1. Run model-forge test suite: `cd model-forge/backend && uv run pytest tests/`
2. Manual test: fit with `Region(30,45,"flat")` + `Region(45,60,"linear")`. Plot. Verify:
   - Flat region has near-zero slope
   - Linear region has near-zero curvature
   - Free regions don't distort (requires baseline anchoring)
3. Join test: C0/C1/C2 at a single point. Verify visual continuity behavior.
4. Verify zero diff to SuperGLM: `cd superglm && git diff HEAD --stat`

### Next Milestone: Baseline Anchoring

The critical remaining feature. Without it, free regions distort when constrained regions are
added. The editing workflow requires:
- Fit baseline `f0(x)` with ordinary Spline
- Refit with `SegmentedSpline(baseline_curve=(grid, f0_values), regions=[...])`
- Free regions stay close to `f0`, edited regions obey constraints
- Visual test: free regions should track the baseline spline closely
