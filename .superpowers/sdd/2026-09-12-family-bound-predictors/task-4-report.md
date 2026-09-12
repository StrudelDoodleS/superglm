# Task 4: documentation, installed typing, and verification

Implementation base: `fdecfe3c` on `work/fresh-0.32.0`.

## Changes and self-review

- Migrated all live constructor examples in README, distributional models,
  distributional inference, and custom-family development to the family-first
  constructor. All parameters are explicit. Examples use root imports for the
  built-in families and declaration helpers.
- Documented family-instance ownership, arbitrary declaration order with
  canonical result order, explicit empty declarations, the missing-predictor
  error with arrows and explanatory ellipses, numeric string semantics, `s`,
  `cat`, `re`, generic `term`, explicit `interaction`, and the two-spline-parent
  interaction-only `ti` semantics. Custom families use `bind_predictor`.
- Kept Tweedie's construction helpers (`mu`, `phi`, `p`) separate from canonical
  names (`mean`, `dispersion`, `power`) used by offsets, predictions, results,
  smoothing keys, and artifacts.
- Added the packaged PEP 561 marker and a reproducible installed-wheel consumer
  check in `scripts/check_installed_typing.py`, with positive and negative
  fixtures under `tests/typing/`. The checker version comes from the existing
  exact dev dependency pin, currently `ty==0.0.72`.
- The consumer check writes the wheel outside the repository, inspects its marker,
  creates an independent Python 3.13 environment, installs the actual wheel
  and its runtime dependencies, removes inherited `PYTHONPATH`/`VIRTUAL_ENV`,
  imports under Python's isolated mode, and asserts that the imported package
  is inside that environment's site-packages. It copies fixtures outside the
  source tree and points ty at that environment. No editable install or source
  override participates in the consumer check.
- Positive `assert_type` checks cover root imports, bound terms, family
  helpers, generic binding, and complete Gaussian/Tweedie construction.
  Negative cases require exactly two unknown-helper and three invalid-argument
  errors at their expected fixture lines. An unknown/Any import fallback cannot
  satisfy these positive and negative checks together.
- Reviewed the complete task diff. No numerical implementation, version field,
  lock pin, persisted schema, controller-owned plan/spec, or historical research
  receipt changed. Generated wheels and environments remain temporary.
  Source searches found no remaining removed public constructor docstrings;
  the internal compiler's `family=`/`predictors=` calls remain internal.

## Completed checks

- Red before green: running the new wheel check before adding `py.typed`
  failed specifically with `AssertionError: wheel lacks PEP 561 marker`.
  Evidence: `/tmp/superglm-task4-typing-red.txt`.
- `uv run python scripts/check_installed_typing.py`: passed after adding the
  marker and again after formatting the checker. Latest installed import was
  `/tmp/superglm-wheel-typing-c3bbu6y8/venv/lib/python3.13/site-packages/superglm/__init__.py`.
  Positive typing passed; all five expected negatives appeared. Wheel contents
  included `superglm/py.typed`. Evidence: `/tmp/superglm-task4-wheel.txt`.
  uv reported a harmless cross-filesystem hardlink fallback to copying.
- Executed all 17 documented `SuperLSS` construction expressions with their
  imports and family/term declarations. All constructed successfully. Also
  executed the complete custom four-parameter example through fitting and
  prediction; its result columns were `a`, `b`, `c`, `d`. This was a construction
  audit, not a claim that all documented real-data fitting examples ran.
- `uv run pytest tests/test_bound_terms.py tests/test_bound_predictors.py
  tests/test_bound_superlss.py tests/test_release_packaging.py -q`: **98 passed
  in 30.78s**. Evidence: `/tmp/superglm-task4-focused.txt`.
- `uv run ty check src/superglm --output-format concise`: **891 diagnostics**,
  matching the supplied pre-change baseline exactly after normalizing line and
  column positions. Diagnostic multiset comparison found **zero additions and
  zero removals**. This is baseline parity, not a clean repository-wide type
  check. Evidence: `/tmp/superglm-task4-ty.txt` and
  `/tmp/superglm-predictor-baseline-ty-ci.txt`.
- `uv run ruff check src/ tests/`: passed.
- `uv run ruff format --check src/ tests/`: passed, 774 files already formatted.
- `uv run ruff check scripts/check_installed_typing.py tests/typing`: passed.
- `uv lock --check`: passed, 147 packages resolved.
- `uv pip check`: passed, 72 installed packages compatible.
- `uv run python run_test.py`: exit 0 and `END-TO-END COMPLETE`. It also printed
  its informational `U-shape: CHECK` label; this script does not assert that
  shape. No claim of a successful U-shape recovery is made here.
- `git diff --check`: passed.

## Full suite and focused correction

`SUPERGLM_REQUIRE_DATA=1 uv run --with mpmath pytest tests/ -q -m "not browser"`
completed with **14,896 passed, 1 failed, 8 skipped, 69 deselected, 186 warnings
in 1618.02s (26m58s)**. The mpmath overlay matches CI. Both local real datasets
were present, and `test_realdata_parity.py`, `test_screening_guide_numbers.py`,
and `test_mixed_interaction_screening.py` completed without dataset skips.
The complete non-browser run includes the non-slow suite as well as slow tests;
it was not repeated under the narrower `not slow` selection. Browser tests
were explicitly deselected and are not claimed as executed coverage.
Evidence: `/tmp/superglm-task4-full.txt`.

The sole failure was
`test_contracts_and_family_adapters_follow_the_one_way_edge_table`. Its
pre-existing import allowlist rejected the new `families._predictors` import
in all eight built-in family adapter modules. A focused reproduction failed
identically before the test correction:
`/tmp/superglm-task4-architecture-red.txt`.

Updated only the explicit architecture policy in that test:

- Each adapter may import `families._predictors` to inherit its statically
  declared public construction helpers.
- `families._predictors` may import `binding` to create declarations and
  `family` for its static protocol annotation.
- `binding` may import `family` for validation and `predictor` for normalized
  templates. It is now itself subject to the edge allowlist.

All other forbidden edges remain forbidden. The helpers and binding cannot
import compiler, solver, or API modules under this check. Kernel isolation,
aggregate-import restrictions, cycle detection, and the prohibition on
solver/API imports of family adapters remain unchanged.

After this test-only correction,
`uv run pytest tests/test_distributional_family_kernel_architecture.py -q`
passed **7 tests in 3.45s**. Ruff check and format check passed for that file.
Evidence: `/tmp/superglm-task4-architecture.txt`. No production code changed
after the full run, so the full suite was not repeated. The original full log
is intentionally retained with its failure; it is not presented as an all-green
single invocation.

## Skips and warnings

A focused `-rs` audit reproduced all eight skips, with **20 passed, 8 skipped,
117 deselected in 3.12s**. Evidence: `/tmp/superglm-task4-skips.txt`.

Four optional R comparisons skip at `tests/_r_harness.py:73`, with the exact
reason `requires R with mgcv and jsonlite`:

- Generalized gamma: `test_location_form_matches_gamlss_gg_at_a_parametric_specification`.
- Generalized Pareto: `test_the_fit_matches_gamlss_gp_at_a_parametric_specification`.
- Log-normal: `test_location_form_matches_gamlss_logno_at_a_parametric_specification`.
- Two-piece: `test_real_line_family_matches_gamlss_sn2_at_a_parametric_specification`.

These tests also require gamlss if the base R harness is available, but this
environment skipped at the earlier base-harness gate. No R/gamlss comparison
is claimed as executed.

Four cases of `test_changed_lookup_storage_or_metadata_declines_range` skip
at `tests/test_group_matrix_range.py:139`, with the exact reason
`The exact spline category lookup needs no permutation`. These are the four
mutation variants (`replace`, `setstate`, `reshape`, `custom`) for the absent
`SplineCategoricalGroupMatrix._row_order` permutation lookup attribute, so the storage-mutation operation
does not apply. They are structural non-applicability skips, not missing data.

The full warning summary was audited by emitting test and source location.
Counts across its categories sum exactly to 186:

| Category | Count | Cause |
| --- | ---: | --- |
| `FutureWarning` | 87 | Legacy `splines=` auto-detection tests and sklearn forwarding exercise the existing deprecation. |
| `SeparationWarning` | 49 | Empty-response categorical cells and overflow-guard termination in screening, quasi-separation, random-effect, robust-solve, and inference fixtures. |
| `FractionalFrequencyWeightWarning` | 19 | Deliberately fractional replication weights in geometry/weight-contract tests and the screening-guide real-book comparison. |
| `PriorWeightLatticeWarning` | 11 | Binomial prior weights and off-lattice Poisson frequency responses in contract, diagnostic, and screening-guide fixtures. |
| `UserWarning` | 8 | Three certifiable-boundary censored power estimates, one unused categorical-level pinning, one collapsed piecewise breakpoint set, and three custom-family Gaussian-shaped scale-profile warnings. |
| `NBThetaBoundWarning` | 6 | Near-Poisson data or profile-publication fixtures hit the configured upper theta search bound. |
| `PerfectSeparationWarning` | 2 | The statsmodels comparator in pandas/Polars IRLS-weight parity detects perfect separation. |
| `RuntimeWarning` | 2 | Intentional extreme-value fixtures overflow `exp` in the finite-log/unrepresentable-factor discretization test and `matmul` in the extreme-scale SCOP Jacobian certification test. |
| `GeneralizedGammaInitializationWarning` | 1 | Infinite-mean adversarial data require shrinking the initial shape toward the log-normal. |
| `GeneralizedParetoInitializationWarning` | 1 | Data beyond the configured shape walls require a usable interior initialization. |

These warnings come from the named existing legacy, boundary, weight-contract,
and adversarial scenarios. The two RuntimeWarnings are unsuppressed warnings
in deliberate overflow tests, not a claim that no numerical warning occurred.
No warning originated in the new declaration helpers, binding, installed typing
check, or migrated constructor parity tests. No new unexplained warning was
identified from this source/context audit; no pre-change full warning-count
baseline was supplied, so count-for-count warning parity is not claimed.
