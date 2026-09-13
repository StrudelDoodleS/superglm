# Tensor support handoff implementation report

Date: 2026-09-13

Worktree: `.worktrees/adaptive-interactions`, branch `research/adaptive-interactions`

The initial implementation was reviewed as a working diff against
`074d999a501b4752fd6c06d52e2be81d7bc01b7c`. The producer-scope amendment below
follows the preliminary measurements of `90f519f39ec3f1a7b1b9a3a2ff13d1c96ababd21`.
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

`_FixedPenaltyInputs` records construction-time input evidence only for a
cache-backed producer. The existing
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

After the producer-scope amendment, the final focused command passed 139 tests
in 8.83 seconds:

```sh
uv run pytest tests/test_penalty_fixed_tensor_reuse.py tests/test_penalty_raw_context_reuse.py tests/test_penalty_algebra_support.py tests/test_distributional_penalty_context.py tests/test_discretize_fit.py::TestDiscretizedTensorInteraction::test_rebuild_design_matrix_freezes_unprojected_tensor_basis tests/test_discretize_fit.py::TestDiscretizedTensorInteraction::test_penalty_context_cache_reuses_frozen_tensor_components tests/test_discretize_fit.py::TestDiscretizedTensorInteraction::test_tensor_pair_summary_cache_reuses_static_marginal_eigenvalues -q
```

The new module includes 60 tests after parametrization. The analytic four-column
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

The mutation driver was rerun after the producer-scope amendment. It patched
functions in process and restored each with
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
| Public width 64 entry family | 0 | 0 | 0 |
| Public width 64 optimizer family | 164,736 | 390,808 | 0 |

Input snapshots include two raw penalties, two solver penalties, the map and
small spectral arrays. Support snapshots include the existing selected support
and its error ledger. Only the optimizer retains fixed input snapshots after
the producer-scope amendment. Validation still temporarily materializes exact
comparison bytes. The target does not copy the immutable solver matrices or
support arrays.

## Cache-backed producer amendment

The preliminary `90f519f3` implementation gave uncached entry families input
snapshots. The entry list remains live through optimization, although
finalization consumes the optimizer's later family. The small width-64 entry
held 164,736 unnecessary bytes, independently of the optimizer's exact evidence.
The parent's preliminary matched measurements showed lower complete-fit times,
exact outputs and unchanged retained model payload, but higher peak RSS.
Those measurements describe the earlier commit and do not establish the
amended candidate's peak-memory behavior.

The approved amendment changes only the provenance-capture guard from
`_can_cache_penalty_group(gm)` to the existing `can_cache_group` flag. This flag
requires a non-`None` component cache, including an initially empty dictionary.
The discrete optimizer is the sole current production caller that constructs
contexts with that cache. Its bootstrap, iteration and final rebuilds all use
`penalty_context_cache`. The uncached entry, EFS, NB-profiling and distributional
builders have no fixed-handoff production consumer.

The supported contract now requires cache-backed source construction. An
uncached context remains numerically usable and can construct its own support,
but cannot later acquire transfer authority. A cache hit never adds new
authority to an existing owner. Targets still consume eligibility, including
fresh fallbacks. The source's exact input, mutation and arithmetic checks do
not change. No new public argument or numerical tolerance was introduced.

Before the guard edit, both the new small allocation regression and the public
ownership regression failed because `initial.fixed_inputs` was non-`None`.
The red run finished in 2.97 seconds. After the one guard edit they pass within
the 139-test focused run. Direct source fixtures now explicitly use `cache={}`.
The new regression also verifies that an ordinary uncached support evaluation
cannot create a receipt, that such an entry cannot authorize a handoff, and
that the cached source still authenticates. The public fit retains support
count one and preserves owner lifetime, predictions and dispersion.

For a float64 tensor of width `p` with component spectrum lengths `r1`, `r2`,
the eliminated live input byte payload is

```text
8 * (5 * p**2 + r1 + r2)
```

The five dense matrices are two raw penalties, two solver penalties and the
coordinate map. The published housing widths and recorded component ranks give:

| Case | Tensor width | Component spectrum lengths | Removed live input bytes | MiB |
| --- | ---: | ---: | ---: | ---: |
| `rows20` | 361 | 342 + 342 | 5,218,312 | 4.97657 |
| `rows30` | 841 | 812 + 812 | 28,304,232 | 26.99302 |

These are exact array-byte savings, excluding small Python-object overhead.
They remove no optimizer support/error evidence and no entry support snapshot,
because the entry support remained lazy. The width-64 byte inspection after
the amendment confirmed zero entry input/support authorization bytes, unchanged
optimizer input/support snapshots of 164,736 and 390,808 bytes, and zero terminal
authorization bytes. Allocator behavior and overlapping work still determine
peak RSS; these payload savings are not a measured peak-RSS reduction.

The first
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
