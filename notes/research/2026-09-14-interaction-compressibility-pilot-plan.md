# Interaction compressibility pilot implementation plan

> **For agentic workers:** Use `superpowers:executing-plans`, or
> `superpowers:subagent-driven-development` for authorized independent work.
> Complete the tasks and their checks in order. This is a plan, not a record
> of implemented code or measured compression.

**Goal:** Determine how much of the saved Airfoil, Concrete and King County
interaction groups can be represented with fewer factors or marginal products.

**Architecture:** A research-only geometry module extracts the actual centered
tensor coefficients and computes two post-fit approximation families. An
isolated worker evaluates every declared candidate on training and development
validation data, preserves the complete reference predictor, and exports
size-error-loss curves and receipts. This batch performs no model fitting.

**Tech stack:** Existing Python 3.13 development environment, NumPy/SciPy,
pytest, Ruff, Matplotlib and the frozen broad-trial adapters/artifacts.

**Spec:** M0/M1 of the
[research plan](2026-09-14-cheap-interaction-research-plan.md).

## Global constraints

- Work in `.worktrees/adaptive-interactions`; preserve unrelated user changes.
- Keep prototypes in `benchmarks`, findings in `docs/research`, and raw runs
  under `.benchmark-artifacts/interaction-compressibility/`.
- Do not edit frozen broad-trial scripts, source files, models or receipts.
  This batch makes no production, dependency, release or API changes.
- Evaluate all retained training/validation rows. Do not use the plotting
  grid or convex-hull mask as a substitute for the observed distribution.
- Keep the reference main effects and intercept fixed in this diagnostic.
  Direct joint refitting belongs to M2 and is a different experiment.
- Every result is post-fit/oracle development evidence, not a numerical
  certificate, a cheap-fitting result or a fresh held-out claim.
- Use at most three concurrent subagents, with mathematical agents at
  `gpt-6-astra` / `max`. Run diagnostic workers serially, one native thread.
- Each worker has a 180-second whole-process deadline; the three-case batch
  has a 900-second cap. Preserve refusal/timeout outputs and reap owned workers.

## Files and ownership

| File to create | Responsibility |
| --- | --- |
| `benchmarks/interaction_compression_geometry.py` | Coefficient extraction, paired scoring, product-metric truncation and modal-product candidates |
| `benchmarks/test_interaction_compression_geometry.py` | Coordinate, metric, reconstruction and adverse-geometry tests |
| `benchmarks/benchmark_interaction_compressibility.py` | Source/model admission, serial workers, complete-predictor comparisons and receipt export |
| `benchmarks/test_interaction_compressibility.py` | Admission, accounting, no-fit scope and candidate completeness checks |
| `benchmarks/interaction_compressibility_cases.json` | Frozen three-case identities and candidate policy |
| `docs/research/2026-09-14-interaction-compressibility.md` | Results, plots, limitations and next-branch recommendation |
| `docs/research/2026-09-14-interaction-compressibility-measurements.json` | Durable derived measurements and raw artifact hashes |

The report/measurement files are created only when the experiment runs. Raw
arrays and standard scientific figures go under the experiment's artifact
directory and a corresponding `docs/research/figures/` directory. A later
execution date may be recorded inside these planned files without rewriting
the frozen source references.

## Task 1. Freeze the three saved models and admit their coordinate maps

**Existing sources to read:**
`case_surfaces` in `docs/research/plot_broad_interaction_surfaces.py`,
`src/superglm/features/interaction.py:1799`,
`benchmarks/benchmark_broad_interactions.py:183`,
`benchmarks/broad_interaction_data.py`, and the broad measurement JSON.
These are input references; do not refactor them for this pilot.

**Fixed cases:**

```json
{
  "schema_version": 1,
  "scope": "post_fit_compression_development_only",
  "run_root": ".benchmark-artifacts/broad-interactions/frozen-20260914",
  "measurement_path": "docs/research/2026-09-14-broad-interaction-measurements.json",
  "measurement_sha256": "45e1934e516b4b2db3aadc0b9072d5b81bc656912c95d6a25aa631fbb942f7fc",
  "cases": {
    "uci_airfoil": "k4_s2",
    "uci_concrete": "k6_s4",
    "kaggle_king_county_sales": "k6_s4"
  },
  "partitions": ["train", "valid"],
  "rank_menu": "all integers from zero through min(left_width,right_width)",
  "modal_menu": "nested marginal prefixes from zero through full width",
  "candidate_scope": ["one_term_at_a_time", "all_terms_common_budget"],
  "fit_allowed": false,
  "test_evaluation_allowed": false
}
```

Read model hashes and exact selected pair order from the admitted measurement
file. Verify its recorded source identity against the current imported package
and frozen runner/adapters before loading local pickle artifacts. Load only
the specified selected models and needed additive controls.

**Geometry interfaces to implement:**

```python
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class RankApproximation:
    coefficients: Array
    singular_values: Array
    discarded_product_energy: float
    realized_product_energy: float
    rank_budget: int
    factor_entries: int


def effective_tensor_coefficients(spec, beta: Array) -> Array:
    """Apply the saved runtime map and return centered marginal coefficients."""
    if spec._decompose:
        raise ValueError("decomposed tensor requires a separately validated map")
    effective = np.asarray(beta, dtype=np.float64).ravel()
    if spec._R_inv is not None:
        effective = spec._R_inv @ effective
    if effective.size != spec._p1 * spec._p2:
        raise ValueError("coefficient count does not match centered marginal widths")
    return effective.reshape(spec._p1, spec._p2)
```

The implementation also rejects unsupported term classes and nonfinite or
incompatible map inputs. Do not treat all model beta arrays as ready to
reshape. For this admitted nondecomposed case, obtain beta by the selected
term's fitted group slice, as the existing plot script does.

- [ ] Write a regression with a nonidentity runtime map and nonsquare target
  matrix. Its expected coefficient matrix uses exact small integer arithmetic;
  dropping the map or transposing the reshape must fail.
- [ ] Add refusals for changed model/source hashes, decomposed terms,
  unsupported families/links, and mismatched coefficient widths.
- [ ] Run the new tests and record that they fail before implementing the
  extraction/admission functions.
- [ ] Implement extraction and admission. Use the saved centered marginal
  ingredients to evaluate paired bases, keeping original feature units in
  the receipt. Score in link units without exponentiation or extra centering.
- [ ] Replay recorded validation loss using the same source/environment and
  compare extracted term scores with `spec.score` on every training and
  validation row. Record discrepancies and an operation-derived roundoff bound
  for equivalent contraction orders; do not use a fixed decimal tolerance.
- [ ] Rerun the focused tests, inspect the diff and commit this bounded unit.

## Task 2. Add product-metric rank and modal-product diagnostics

**New interfaces:**

```python
def score_pairs(left_basis: Array, coefficients: Array, right_basis: Array) -> Array:
    """Evaluate rows b_left[i].T @ C @ b_right[i] with bounded row workspace."""
    return np.einsum("ij,jk,ik->i", left_basis, coefficients, right_basis)


def truncate_in_product_metric(
    coefficients: Array,
    left_metric_factor: Array,
    right_metric_factor: Array,
    rank: int,
) -> RankApproximation:
    """Diagnostic SVD with M_left=R_left.T R_left and similarly on the right."""


def modal_prefix(
    coefficients: Array,
    left_modes: Array,
    right_modes: Array,
    width: int,
) -> Array:
    """Keep nested prefixes of the two marginal penalty eigensystems."""
```

These last two declarations specify interfaces; their algorithms and test
oracles are below. They are not completed Python implementations.

For observation weights `w`, normalize `a=w/sum(w)`. Use only training bases
to build metric factors, for example thin QR of `sqrt(a)[:,None] * B_left`.
This represents the product of the two weighted empirical marginal measures,
not the empirical joint distribution. Record factor reconstruction error,
conditioning estimates and factorization failures. An ill-conditioned or
rank-deficient factor is a diagnostic refusal for this first implementation;
do not silently invert it, add a ridge, or claim a certified support quotient.
The refusal criterion must use an explicit operation/conditioning error budget.

For full-rank upper triangular factors, the algorithm is:

```python
from scipy.linalg import solve_triangular

whitened = left_metric_factor @ coefficients @ right_metric_factor.T
left, singular_values, right_t = np.linalg.svd(whitened, full_matrices=False)
kept = (left[:, :rank] * singular_values[:rank]) @ right_t[:rank]
partial = solve_triangular(left_metric_factor, kept, lower=False)
candidate = solve_triangular(right_metric_factor, partial.T, lower=False).T
discarded_product_energy = float(singular_values[rank:] @ singular_values[rank:])
residual = left_metric_factor @ (coefficients - candidate) @ right_metric_factor.T
realized_product_energy = float(np.linalg.norm(residual, ord="fro") ** 2)
```

Keep input admission, error reporting and the `RankApproximation` construction
around this kernel. The tail is a floating-point evaluation of the exact
product-metric identity, not an outward certified error bound.
Report both the spectral tail and the realized recovered-matrix error.
A zero full-rank tail does not imply an exactly reconstructed candidate.
Check their discrepancy against an allowance derived from SVD reconstruction
and orthogonality residuals, the triangular recovery residual, and the
arithmetic used to evaluate those quantities. The diagnostic must label
estimated allowances separately from any outward-certified enclosure.

This well-conditioned test checks which direction the metric favors. The
weighted rank-one optimum has squared error 4; keeping the unweighted leading
direction has error 9. Their midpoint supplies a decision threshold, not a
floating-point accuracy tolerance:

```python
def test_rank_budget_uses_function_metric():
    coefficients = np.diag([2.0, 1.0])
    left_factor = np.diag([1.0, 3.0])
    right_factor = np.eye(2)
    result = truncate_in_product_metric(coefficients, left_factor, right_factor, rank=1)
    residual = left_factor @ (coefficients - result.coefficients) @ right_factor.T
    realized_error = float(np.linalg.norm(residual, ord="fro") ** 2)
    decision_threshold = (4.0 + 9.0) / 2.0
    assert realized_error < decision_threshold
    assert result.factor_entries == 4
```

This detects the wrong retained direction if whitening is removed. Add a
separate accuracy test using the operation-derived reconstruction allowance;
the decision test alone does not verify rounding accuracy. Do not require
bitwise SVD/recovery outputs even for a representable mathematical answer.
Degenerate and near-rank fixtures assert reconstruction/backward error,
metric refusal and stable observables, without fixing singular-vector signs.

For modal prefixes, use the saved centered marginal penalty matrices and
orthogonal eigensystems, ordered by nondecreasing roughness. Transform
`D=U_left.T @ C @ U_right`, retain the first `min(width,k_left)` rows and
`min(width,k_right)` columns, then map back with `U_left @ D_kept @ U_right.T`.
The full prefix reconstructs the reference. This is an oracle coefficient
truncation; it is not a Galerkin optimum or a claim of best product selection.
No eigenvalue clipping may silently change the source penalty. Ambiguous
numerical nullity is recorded and prevents a nullspace-certification claim.
For repeated eigenvalues, freeze and record the returned marginal mode bases
once per admitted reference. Every prefix uses those same bases. Prefixes
cutting a tied eigenspace depend on that choice, so their curves are not
claimed invariant under rotations within the tied space.

- [ ] Add the metric test above, rank-zero/full-rank cases, and a singular
  metric refusal before implementing the new functions.
- [ ] Add an exact correlated-pair counterexample: a coefficient change can
  have nonzero product-measure norm and zero error on observed diagonal pairs.
  The report must keep those norms in separate fields.
- [ ] Add a reconstruction test under invertible changes of marginal
  coordinates. At a separated singular-value cutoff, compare reconstructed
  predictions and metric errors with derived rounding bounds. At a repeated
  cutoff, compare achieved metric error and rank; equally optimal truncations
  may have different predictions. Never compare singular-vector signs.
- [ ] Add nested modal-prefix tests and an adverse case whose useful component
  lies beyond the first coarse prefix. This prevents coarse adequacy being
  assumed from a small coarse score.
- [ ] Demonstrate the direction-choice test fails if whitening is deleted and the
  coordinate test fails if the runtime map is deleted.
- [ ] Implement the admitted kernels, rerun tests and Ruff, then commit.

## Task 3. Evaluate complete saved predictors and export the evidence

**Consumes:** admitted selected models, per-term coefficient matrices, bases,
rank/modal candidates and the fixed case manifest. Use the existing
`run_isolated` worker owner pattern and `retained_model_storage` accounting
where applicable; avoid building a second process-management framework.

**Produces:** per-attempt JSON and NPZ arrays, one aggregate measurement JSON,
standard scientific size-error-loss figures, and a research report.

For term set `J`, evaluate the modified predictor as

```python
candidate_prediction = reference_prediction.copy()
for name in replaced_terms:
    candidate_prediction += candidate_effect[name] - reference_effect[name]
```

Do this first for one term at a time, then for all selected terms with a common
rank or prefix budget, clipped to each term's feasible dimensions. Preserve
the full predictor's main effects and intercept. Full-rank/full-prefix
reconstruction must recover its loss within the derived evaluation bound.
Rank zero removes the interaction contributions but leaves mains at their
joint-reference estimates; it is not the separately fitted additive model.

Load the saved matching-k and validation-best additive controls separately
when reporting gain retention. Their hashes and validation losses must replay
too. The production procedure can choose a larger additive parent space later;
this pilot does not settle that statistical comparison.

Each attempt records:

- Case/model/source/input hashes; training/validation row identities; term
  order, marginal widths and actual rank or selected product count.
- Original coefficient entries, factor entries, diagnostic array payload and
  candidate representation payload separately. An explicitly reconstructed
  candidate matrix is diagnostic workspace, not a compact deployed model.
- Product-measure squared error, paired weighted prediction error, maximum
  paired discrepancy, and full-predictor training/validation loss.
  Store spectral tail and realized product error in separate fields and keep
  per-edge product errors distinct from aggregate paired-prediction error.
- One-term/all-term scope, reference and additive loss, and gain-retention
  fraction only when its denominator is meaningful.
- Conditioning/rounding diagnostics, refusal reason, elapsed diagnostic
  worker time, process high-water RSS and the historical reference-fit cost.
- `numerical_certificate: false`, `new_model_fits: 0`,
  `fresh_test_evaluation: false`, and `fitting_speedup_claim: false`.

- [ ] Write failing tests that all requested rank/prefix budgets and refusals
  reach the aggregate receipt, including a case with no admissible candidate.
- [ ] Add a scope test that replaces `SuperGLM.fit` and `fit_reml` with raising
  stubs during the diagnostic worker. No path may learn parameters anew.
- [ ] Add a receipt test distinguishing the rank-zero modified predictor from
  the separately fitted additive control, and candidate factor payload from
  the diagnostic's expanded coefficient storage.
- [ ] Implement the worker and aggregate export. Read only train/validation
  outcomes; retain the old test designation as already-used development data.
- [ ] Run focused checks, then execute the three-case batch serially:

```bash
uv run pytest benchmarks/test_interaction_compression_geometry.py benchmarks/test_interaction_compressibility.py -q
uv run ruff check benchmarks/interaction_compression_geometry.py benchmarks/benchmark_interaction_compressibility.py benchmarks/test_interaction_compression_geometry.py benchmarks/test_interaction_compressibility.py
uv run ruff format --check benchmarks/interaction_compression_geometry.py benchmarks/benchmark_interaction_compressibility.py benchmarks/test_interaction_compression_geometry.py benchmarks/test_interaction_compressibility.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 uv run --no-sync python benchmarks/benchmark_interaction_compressibility.py --manifest benchmarks/interaction_compressibility_cases.json --output .benchmark-artifacts/interaction-compressibility/pilot-20260914
```

The planned runner exposes `--manifest` and `--output`, refuses an existing
output directory and owns its serial subprocesses. These commands become
available when Tasks 1–3 are implemented; they have not run for this plan.

- [ ] Independently replay array-derived losses and verify all receipt hashes.
  Inspect the figures and document every metric refusal and adverse outcome.
- [ ] Write the result and next-branch recommendation, run `git diff --check`,
  and commit the checked pilot. No full production test rerun is claimed for
  this research-only batch unless it changes an existing production path.

## Completion and next decision

The pilot is complete when the three saved model identities replay, all ten
selected terms have a complete attempted-candidate record, the coordinate and
metric tests pass with their mutation demonstrations, and the report shows
component and aggregate size-error-loss curves with all refusals visible.
A completed pilot may find no useful compression.

There is no Lean implementation requirement for this empirical diagnostic.
The report cites the exact product-metric identity and labels the computed
tail as a numerical diagnostic. The P1/P2/P3 proof and arithmetic obligations
in the main research plan become necessary before M2 claims a certified
fixed-target reduction. Existing Lean identities do not close those gates.

Before choosing a production representation, extend the diagnostic to a
verified wide California housing coefficient snapshot, whose acquisition is
a separately charged bounded task. Then select the next branch using the
observed mechanism and write its direct-fitting specification. These tiny
k=4/6 models cannot establish rich-interaction scaling on their own.
