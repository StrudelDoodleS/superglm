# Generator Status — Round 1

## Requirements Self-Assessment

### Feature A: Line+markers for categorical relativities
| Req | Description | Status |
|-----|-------------|--------|
| R-A1 | Ordered categoricals: `go.Scatter(mode="markers")` + error bars + smooth curve | PASS |
| R-A2 | Unordered categoricals: `go.Scatter(mode="lines+markers")` + error bars | PASS |
| R-A3 | Removed `display_mode`/`show_bars`/`show_markers` branching | PASS |
| R-A4 | `_resolve_categorical_display()` retained, no longer gates rendering | PASS |
| R-A5 | Link-scale toggle works correctly | PASS |

### Feature B: Exposure bars on dual y-axis
| Req | Description | Status |
|-----|-------------|--------|
| R-B1 | Absolute sample_weight sums (no normalization) | PASS |
| R-B2 | Exposure on yaxis="y3" (right y-axis) with opacity ~0.3 | PASS |
| R-B3 | Density traces before term traces (z-order behind) | PASS |
| R-B4 | `needs_lower_panel` guard removed | PASS |
| R-B5 | `_XAxisConfig` sets `overlay_density_top=True` | PASS |
| R-B6 | yaxis3 autorange, title="Exposure" | PASS |
| R-B7 | Hovertemplate shows absolute exposure | PASS |

### Feature C: show_knots for OrderedCategorical
| Req | Description | Status |
|-----|-------------|--------|
| R-C1 | `term_inference()` populates `spline=` for OrdCat spline-basis | PASS |
| R-C2 | `_supports_spline_diagnostics()` accepts categoricals with spline | PASS |
| R-C3 | `_add_spline_diagnostic_traces()` uses `ti.smooth_curve` for OrdCat | PASS |
| R-C4 | Categorical path wires `_add_spline_diagnostic_traces()` call | PASS |

### Feature D: collapse_levels + grouping
| Req | Description | Status |
|-----|-------------|--------|
| R-D1 | `LevelGrouping` frozen dataclass with required fields | PASS |
| R-D2 | `collapse_levels()` supports from_level, below, groups | PASS |
| R-D3 | Validation: duplicates, unknown levels, mixed modes | PASS |
| R-D4 | `OrderedCategorical(grouping=...)` integration | PASS |
| R-D5 | `Categorical(grouping=...)` integration | PASS |
| R-D6 | `term_inference()` expands grouped levels | PASS |
| R-D7 | Public API exports | PASS |

### Cross-cutting
| Req | Description | Status |
|-----|-------------|--------|
| AC-X1 | Full test suite passes | PASS (1628 passed, 34 skipped) |
| AC-X2 | Lint clean | PASS |
| AC-X3 | No regressions in existing tests | PASS |

## Tests Added/Modified

### New: `tests/test_categorical_ux.py` (34 tests)
- **TestFeatureA** (7 tests): no bar traces, markers present, smooth curve, lines+markers, no text traces, link variants, ordered ordered-cat
- **TestFeatureB** (7 tests): absolute exposure, y3 axis, z-order, opacity, autorange, hovertemplate, bar name
- **TestFeatureC** (4 tests): spline metadata, diamond knots, show_knots=False, step basis
- **TestCollapseLevels** (9 tests): from_level, below, groups, combined, duplicate membership, unknown levels, mixed modes, identity mapping, all_original_levels
- **TestOrderedCategoricalGrouping** (3 tests): fit/predict, expanded levels, original level predict
- **TestCategoricalGrouping** (3 tests): fit/predict, expanded levels, original level predict
- **TestGroupingPlot** (1 test): smoke test for grouped model plotting

### Modified: `tests/test_plot_api.py`
- Updated 7 tests from `go.Bar` expectations to `go.Scatter` for categorical relativities
- Updated density-related tests from "Exposure density" to "Exposure"
- Updated style override test for Numeric (still uses `go.Bar`)

## Files Changed

### New files
- `src/superglm/features/grouping.py` — `LevelGrouping` dataclass + `collapse_levels()` function
- `tests/test_categorical_ux.py` — 34 tests for all four features

### Modified files
- `src/superglm/__init__.py` — export `LevelGrouping`, `collapse_levels`
- `src/superglm/features/__init__.py` — export `LevelGrouping`, `collapse_levels`
- `src/superglm/features/categorical.py` — `grouping=` parameter, build/transform mapping
- `src/superglm/features/ordered_categorical.py` — `grouping=` parameter, level value computation, build/transform mapping
- `src/superglm/inference.py` — `_expand_grouped_term()`, spline metadata for OrdCat, grouping expansion
- `src/superglm/plotting/main_effects_plotly.py` — Features A (scatter), B (exposure bars), C (show_knots)
- `tests/test_plot_api.py` — regression fixes for bar→scatter migration

## Known Gaps or Deferred Items
- **T-D11** (single-group edge case): Not explicitly tested. `collapse_levels` handles it but no dedicated test.
- Matplotlib engine: out of scope per PLAN.md, not modified.

## Existing Tests
All 1628 existing tests pass. No regressions.
