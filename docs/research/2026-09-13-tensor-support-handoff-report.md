# Tensor support handoff implementation report

Date: 2026-09-13

Worktree: `.worktrees/adaptive-interactions`, branch `research/adaptive-interactions`

Reviewed working diff against `074d999a501b4752fd6c06d52e2be81d7bc01b7c`.
This worker created no commits. The parent owns integration review, broader
suites and serial housing measurements.

The bounded implementation removes the second selected-support construction
from a small public Gaussian tensor fit. It preserves the existing numerical
support and error ledger. This report establishes focused correctness and
dispatch evidence. It does not claim a complete-fit speedup or successful
housing memory gate.

## Change and ownership

`optimize_discrete_reml_cached_w` now returns its final `penalties` family through
the existing `REMLResult.reml_penalties` field. `fit_ops` forwards that family
when present. The entry family has not acquired the expensive tensor support
and cannot substitute for the optimizer's populated family.

`build_penalty_context` and `build_penalty_components` accept the private
`_reuse_fixed_from` argument. The admission path applies only to complete
unprojected discrete tensor families with the original fixed group cache key.
The existing `_reuse_raw_from` route and its tensor exclusion remain intact.

`_FixedPenaltyInputs` records construction-time input evidence. The existing
`_RawPenaltyFamilyReceipt` owns the support snapshot. `get_support` captures that
snapshot only during its already required first support construction, after
checking the original solver matrices and arithmetic both before and after the
call. It never forces an entry context to compute support. Admission compares
the source and target against the original input record, so a mutation before
the lazy support build cannot become newly trusted at optimizer return.

The target receives new component descriptors and a new mutable geometry owner.
Its authenticated immutable solver matrices, spectra and selected support share
the source's arrays. Weighted evaluations, zero-face state and volume caches do
not transfer. The target consumes handoff eligibility, including when admission
fails and the normal builder constructs a fresh target. It cannot authorize a
second transfer. Pickle and deepcopy also drop both pending input evidence and
completed receipts.

After successful terminal-context construction, finalization replaces the
result's family with the terminal family. A public-fit weak-reference regression
confirms collection of the optimizer geometry owner and component descriptors.
The terminal model keeps its selected support and error arrays without keeping
the duplicate authorization snapshots.

## Exact admission boundary

The fixed route checks:

- Original group-matrix and marginal-basis object identities, tensor ID and
  marginal column widths, and the existing fixed group cache key.
- Ordered raw and solver matrix values, dtypes, shapes and strides, and the
  solver coordinate map with the same exact evidence.
- Component names, group and coefficient placement, penalty kind, repeat/block
  metadata, component types, lambda policies, component ranks, spectra and
  determinant summaries.
- Current source solver/spectral arrays and the selected-support arrays,
  including their backing owners, remain read-only. A read-only view over a
  writable array or byte buffer is insufficient.
- Selected support fields and all existing reconstruction, projection and
  root-error arrays. The independent `_basis_gram_evidence` memo is excluded
  because it retains its own token and validation contract.
- The existing `_raw_penalty_arithmetic` precision, helper-identity and shared
  rank-policy evidence.

Evidence uses exact byte comparisons. There is no digest-only acceptance.
Layout and backing-owner checks are explicit options on the existing helpers;
the raw-coordinate route keeps its previous defaults.

Observation-side `B_joint`/`B_unique` values, marginal evaluation rows, row/bin
indices and observation weights do not define the fixed penalty support. They
are outside this receipt and require their own fresh data geometry. The fixed
map, penalty family and marginal-column identities must still match. This is
a penalty-support handoff, not authorization to reuse a weighted data Gram.

Positive lambda changes can reuse the common unweighted support. Each weighted
summary starts fresh. Zero-face rank continues through the existing active
component filter; no union rank is substituted for a face rank. Changed targets
fall back to ordinary construction. A candidate whose new map cannot preserve
its declared penalty geometry still raises the existing error, and does not
replace the source's accepted evaluation.

## Red, green and mutation evidence

The first effective public regression used 260 observations, Gaussian errors,
two `Spline(n_knots=5)` mains, one discrete tensor and three REML iterations.
On the unfixed implementation the real `_penalty_support` wrapper recorded
`[64, 64]`, failing the expected `[64]`. The independent ownership regression
failed because the real optimizer's `reml_penalties` field was `None`.
Both failed in the same 2.64-second run before production edits.

The original Poisson variant was ineffective for the count assertion: it built
support once because fixed dispersion did not require terminal nullity.
Gaussian exercises the consumer observed in the housing profile.

The final focused command passed 138 tests in 6.02 seconds:

```sh
uv run pytest tests/test_penalty_fixed_tensor_reuse.py tests/test_penalty_raw_context_reuse.py tests/test_penalty_algebra_support.py tests/test_distributional_penalty_context.py tests/test_discretize_fit.py::TestDiscretizedTensorInteraction::test_rebuild_design_matrix_freezes_unprojected_tensor_basis tests/test_discretize_fit.py::TestDiscretizedTensorInteraction::test_penalty_context_cache_reuses_frozen_tensor_components tests/test_discretize_fit.py::TestDiscretizedTensorInteraction::test_tensor_pair_summary_cache_reuses_static_marginal_eigenvalues -q
```

The new module includes 59 tests after parametrization. The analytic four-column
fixture has positive spectrum `(2, 3, 5)` at lambdas `(2, 3)`. It verifies rank 3,
log determinant `log(30)`, log-lambda gradient `(7/5, 8/5)` and diagonal curvature
`6/25`, with negative cross curvature. It checks the selected projector and
reconstruction against dimension/epsilon allowances and the existing error
ledger. Active-face rank is 2 and the all-zero rank is 0. A separate public fit
comparison checks predictions and dispersion with a bound derived from the
final Hessian's dimension, conditioning and dtype epsilon.

Six in-process mutations were each rejected by an existing regression:

| Mutation | Regression that failed under the mutation |
| --- | --- |
| Always accept the support receipt | Mutated `Q_plus` evidence |
| Erase raw-family input evidence | In-place raw matrix change |
| Always accept construction-time solver inputs | Solver changed for support construction, then restored |
| Erase the arithmetic token | Changed support helper after support construction |
| Erase component-summary evidence | Changed declared component rank |
| Use union rank for an active face | One-component face nullity |

The mutation driver patched functions in process and restored each with
`pytest.MonkeyPatch.context`; it did not edit production files. All six printed
`Mutation killed` and the final driver completed successfully. Additional red
runs exposed retained target receipts, writable spectra, writable array backing
owners and read-only support views over mutable byte buffers; each regression
passed after its corresponding correction.

Ruff check, Ruff format check on all five changed Python files, and
`git diff --check` passed. The environment was synchronized with
`uv sync --python 3.13 --extra dev`. No full test suite or housing fit was run by
this worker. Numerical tests paused during the parent's baseline interval.

## Memory cost and remaining gates

Exact evidence has a fit-phase cost. Unique byte-payload counting, without Python
object overhead, measured these additional fixed-handoff authorization bytes:

| Fixture/context | Construction inputs | Support snapshot | Terminal authorization |
| --- | ---: | ---: | ---: |
| Analytic width 4 | 672 | 1,048 | 0 |
| Public width 64 entry family | 164,736 | 0 | 0 |
| Public width 64 optimizer family | 164,736 | 390,808 | 0 |

Input snapshots include two raw penalties, two solver penalties, the map and
small spectral arrays. Support snapshots include the existing selected support
and its error ledger. Entry and optimizer input snapshots can overlap in
lifetime. Validation also temporarily materializes exact comparison bytes.
The target does not copy the immutable solver matrices or support arrays.

These small-fixture payload counts do not establish a lower peak RSS. The first
support construction retains its original asymptotic cost, and authorization
adds transient storage. The parent must compare complete-fit time, fit/process
peak RSS, retained model payload, numerical outputs and actual dispatch on the
matched `rows20` and `rows30` workers before accepting the performance claim.

Changed files are `src/superglm/reml/penalty_algebra.py`,
`src/superglm/reml/discrete.py`, `src/superglm/model/fit_ops.py`,
`src/superglm/model/reml_finalize.py`, `tests/test_penalty_fixed_tensor_reuse.py`,
the [implementation plan](2026-09-13-tensor-support-handoff-plan.md), and this
report. No version files changed. A future PR's advisory impact is `release:patch`
because this preserves numerical behavior while removing duplicate work.
