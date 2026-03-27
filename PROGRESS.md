# Harness Progress

## Task
Categorical UX improvements: line+markers for relativities, exposure bars on dual y-axis, show_knots for OrderedCategorical, and collapse_levels grouping. Read PLAN.md for full spec.

## Status
Round: 1 / 3
Verdict: PENDING (awaiting reviewer)

## Completed

### Feature A: Line+markers for categorical relativities
- [x] R-A1: Ordered categoricals render as `go.Scatter(mode="markers")` with error bars + smooth curve
- [x] R-A2: Unordered categoricals render as `go.Scatter(mode="lines+markers")` with error bars
- [x] R-A3: Removed `display_mode`/`show_bars`/`show_markers` branching
- [x] R-A4: `_resolve_categorical_display()` retained but no longer controls rendering
- [x] R-A5: Link-scale toggle works correctly

### Feature B: Exposure bars on dual y-axis
- [x] R-B1: Exposure bars show absolute sample_weight sums (no peak normalization)
- [x] R-B2: Exposure bars on yaxis="y3" (right y-axis) with opacity ~0.3
- [x] R-B3: Density traces added BEFORE term traces (z-order: behind)
- [x] R-B4: `needs_lower_panel` guard removed for categorical density
- [x] R-B5: `_XAxisConfig` sets `overlay_density_top=True`
- [x] R-B6: yaxis3 uses `autorange=True`, title="Exposure"
- [x] R-B7: Hovertemplate shows "Exposure: %{y:,.0f}"

### Feature C: show_knots for OrderedCategorical
- [x] R-C1: `term_inference()` populates `spline=` for OrderedCategorical spline-basis terms
- [x] R-C2: `_supports_spline_diagnostics()` returns True for categoricals with `ti.spline is not None`
- [x] R-C3: `_add_spline_diagnostic_traces()` handles OrderedCategorical (uses `ti.smooth_curve` when `ti.x is None`)
- [x] R-C4: Categorical path calls `_add_spline_diagnostic_traces()` when `show_knots=True` and `ti.spline is not None`

### Feature D: collapse_levels + grouping
- [x] R-D1: `LevelGrouping` frozen dataclass with required fields
- [x] R-D2: `collapse_levels()` supports `from_level`, `below`, `groups` modes
- [x] R-D3: Validation: duplicate membership, unknown levels, mixed modes
- [x] R-D4: `OrderedCategorical(grouping=...)` works with build/transform
- [x] R-D5: `Categorical(grouping=...)` works with build/transform
- [x] R-D6: `term_inference()` expands grouped levels back to originals
- [x] R-D7: `LevelGrouping` and `collapse_levels` exported from `superglm`

## Tests
- 1628 passed, 34 skipped, 0 failed
- Lint: clean (ruff check src/ tests/)
- 34 new tests in `tests/test_categorical_ux.py`
- Existing tests in `tests/test_plot_api.py` updated for bar→scatter changes

## Outstanding
None — all requirements implemented.

## Round History

### Round 1
- Implemented all four features (A, B, C, D)
- 8 commits on feature branch
- Fixed double-mapping bug in OrderedCategorical grouping
- Fixed yaxis3 autorange in initial layout setup
- Updated existing test expectations for bar→scatter migration
