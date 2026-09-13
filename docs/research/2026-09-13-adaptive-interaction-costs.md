# Adaptive interaction costs and the tensor-support handoff

Date: 2026-09-13. Audited base: `7c4e70ffac99c9a70adf90e5b21d9735109c4675`,
SuperGLM 0.33.0. This note preserves the read-only cost investigation for the
[research design](2026-09-13-adaptive-interactions-research-design.md).
Source pointers below are relative to `src/superglm/` at that base.

The source inspection and archived profile establish the current allocation
paths and repeated work. The context transfer is an unimplemented performance
hypothesis. The operator costs are conditional models, not measured adaptive
fit results or numerical guarantees. No new fits or production changes were
made for this note.

The [baseline receipt](../../benchmarks/adaptive_interactions_baseline.json)
records two uninstrumented observations and one separate profile. All use
12,384 training rows, eight ordinary spline groups and one geographic tensor
group. `rows20` and `rows30` denote geographic knot counts with row-frequency
quantiles, not row counts.

| Run | Coefficients excluding intercept | Complete fit, seconds | Whole-worker peak RSS, MiB |
| --- | ---: | ---: | ---: |
| `rows20`, unprofiled | 513 | 10.053859207953792 | 565.0390625 |
| `rows30`, unprofiled | 1,013 | 117.27918661799049 | 900.609375 |
| `rows30`, profiled | 1,013 | 116.54104198300047 | 906.5390625 |

The tensor has 361 or 841 coefficients. All runs used the `gram` backend and
matched the committed validation/test predictions exactly. These are individual
observations without an estimate of timing variability. Exact local replay is
not a portable numerical tolerance. The fit clock excludes imports, loading,
prediction and export; the RSS high-water mark includes the whole worker through
prediction/export. Retained model memory and fit-phase peak RSS were not
separately measured. The historical split is a performance and numerical
workload, not fresh evidence after model selection.

The `rows30` profile contains the following nested costs. Cumulative times
overlap and must not be added as separate phases.

| Function | Calls | Cumulative seconds |
| --- | ---: | ---: |
| `_penalty_support` | 98 | 103.59241660800001 |
| `_component_root` | 100 | 24.687443652000002 |
| `_group_penalty_rank` | 207 | 103.336641512 |
| `compute_total_penalty_rank` | 1 | 51.601030687000005 |
| `compute_penalty_nullity` | 22 | 51.73677076600001 |
| `finalize_reml_fit` | 1 | 55.433370198000006 |
| `optimize_reml_best` | 1 | 60.541236165 |

Because this model has only singleton and two-component families, 98 support
builds and 100 component-root builds imply 96 singleton support builds and two
tensor support builds. The counts do not describe 98 expensive tensor builds.

The first expensive producer is `reml/discrete.py:446`.
`compute_total_penalty_rank(penalties)` calls `_group_penalty_rank`, which reaches
`geometry.get_support()` at `reml/penalty_algebra.py:650`. The optimizer retains
that support through subsequent positive lambda updates. Finalization builds
another family at `model/reml_finalize.py:404`. Its terminal objective reaches
`reml/objective.py:310`, then penalty nullity and the same support producer.
The profile attributes 51.818442683 seconds to the terminal objective and only
0.871059251 seconds to the optimizer's 20 objective calls. This caller evidence
locates the second expensive build at the finalization boundary.

The first hypothesis is to carry the optimizer-owned positive tensor family
through an explicit fit-local result or workspace and consume its selected
support during finalization. The `REMLResult.reml_penalties` field already
exists, but the discrete return at `reml/discrete.py:1556` does not populate it.
`model/fit_ops.py:2020` instead passes the original entry family. Forwarding that
original family cannot recover the costly optimizer-owned support.

Existing reuse must remain part of the design:

- `dm_builder.py:1225` keeps the same unprojected multi-penalty discrete tensor
  group across lambda changes.
- `reml/discrete.py:276` owns a fit-local cache. Component reuse occurs at
  `reml/penalty_algebra.py:1679`, with keys defined at `:1231`. Thus the lambda
  loop already reuses complete tensor components and their support.
- `_reuse_raw_from` is an existing validated transfer, but `:1781` excludes
  discrete tensors. The proposed change concerns fixed solver coordinates and
  is not a proposal to introduce that raw route again.
- `_rebind_penalty_context` and `_snapshot_penalty_context` at `:807` and `:897`
  already separate mutable owners while sharing selected support. Their dense
  copies cost memory; their existence does not prove that they fit this handoff.

The transfer must check ordered matrix values, dtype and shape, component names
and policies, coefficient placement, and the solver coordinate map. It must
preserve selected rank, support, reconstruction/projection bounds, precision
assumptions and arithmetic/rank-policy evidence. Current cache keys use
identities; identity alone does not detect in-place mutation. Exact evidence
snapshots require comparison work and retained bytes. The existing raw receipts
at `reml/penalty_algebra.py:446` illustrate that cost.

| Change | Reuse consequence |
| --- | --- |
| Positive lambda values change | Common unweighted support remains valid; a weighted summary needs its own exact lambda match. |
| A lambda becomes zero | The active support may change. Do not transfer the positive union rank or weighted summary as a face result. |
| Observation weights change | Weighted data geometry changes. Penalty support may survive only if the penalty target and coordinates remain unchanged, including any weight-dependent basis construction. |
| Ordered matrices, dtype, placement, map or numerical policy change | Refuse the fixed-context transfer and construct fresh evidence. |
| A candidate fails validation | Keep the accepted source owner intact; release the failed candidate's state. |

Zero faces need separate treatment. `_structural_active_penalty_rank` at
`reml/penalty_algebra.py:2134` filters inactive components. `_context_geometry` at
`:752` rejects incomplete families, so a one-component face can construct a
new support. Preserve that existing behavior in the first pilot. Passing the
tensor spectral `support_rank` to all rank consumers would require a separate
proof that it preserves the selected numerical rank contract. Keep the proposed
handoff within one fit rather than extending it across fits or serialization.

For the complete-fit model, let `n` be rows, `p` active coefficients,
`P_max` virtual rich coefficients and `q` smoothing parameters. If row `i`
overlaps `a_i` active basis functions, define `z = sum_i a_i` and
`o = sum_i a_i^2`. Let `e` count stored penalty nonzeros and `t` denote
coordinate-transform application work. Under sparse local evaluation and
structured transforms throughout, one Hessian action costs

\[
A = O(n+p+z+e+t).
\]

A dense transform makes `t = O(p^2)`. The explicit `p` term may be omitted only
under an assumption such as `p <= z+e+t`. Let `C_S` bound one component-penalty
action. Let `C_K` include `A`, a preconditioner application, and any recycling
or orthogonalization work per Krylov iteration. Preconditioner setup remains
a separate cost.

| Work | Cost to charge |
| --- | --- |
| Sparse Gram arithmetic | `O(n+p+o)`, before sparse indexing, coordinate changes and factor fill |
| Dense factorization | `O(p^3)` work and `O(p^2)` storage |
| Sparse factorization | Actual symbolic/numerical fill and factor storage; sparse input alone gives no linear bound |
| Coefficient iterations | `sum_j [C_prec,j + K_j C_K,j]`, including each required setup/update |
| Gradient trace estimates | `O(b_g K C_K + b_g q C_S)` |
| Full outer-Hessian trace estimates | One shared-probe construction costs `O(b_h (q+1) K C_K + b_h q C_S + b_h q^2 p)` |
| Log determinant estimates | `O(b_log ell A)`, plus reorthogonalization and any preconditioner determinant correction |
| Outer Newton step | `O(q^3)` solve and `O(q^2)` storage, plus coefficient-response work |
| Requested inference | Required solves and outputs; a full dense covariance has `Omega(p^2)` output size |

Here `b_g`, `b_h`, and `b_log` count probes, `ell` counts Lanczos steps, and
`K_j` counts Krylov iterations. The table's common `K` abbreviates the relevant
solve counts; measurements must retain their actual distribution. The
outer-Hessian row matters: `b K` solves plus `b q` penalty actions account for
gradient traces, not a general full Hessian. Fixed-weight Gaussian models avoid
the extra weight-derivative work required by general GLMs.

For `R` refinements and `E_r` outer evaluations at refinement `r`, charge

\[
T_{\mathrm{complete}}=
\sum_{r=0}^{R}\left[
 C_{\mathrm{build},r}+C_{\mathrm{search},r}+C_{\mathrm{cert},r}
 +\sum_{e=1}^{E_r}
  \left(C_{\mathrm{coefficients},re}+C_{\mathrm{outer},re}\right)
\right]
+C_{\mathrm{terminal}}+C_{\mathrm{requested}}.
\]

Count rejected lambda trials and basis proposals in the corresponding sums.
Count terminal refits explicitly. Krylov counts depend on conditioning and the
requested error; probe and Lanczos counts depend on accuracy and confidence.
Neither is a fixed constant established by one small fit. Probability budgets
must cover repeated adaptive evaluations. The direct baseline receives its
existing reuse benefits, including reusable fixed-weight Gaussian data
geometry. The crossover is between these complete totals, not between one
Hessian action and one dense factorization.

The present tensor path has several dense allocations:

- `features/interaction.py:1768` creates observed-pair `B_joint`; `:1772`
  creates dense Kronecker penalties. For `s` observed pairs and tensor width
  `p_g`, these require `O(s p_g)` and `O(p_g^2)` storage respectively.
- `_group_matrix/_group_matrix_discretized.py:387` retains `B_joint`. Its
  adjoint at `:495` creates a full marginal-bin grid. Dense `R_inv` applications
  remain at `:480`, `:483` and `:498`.
- Generic penalty matvec at `reml/penalty_algebra.py:1028` ordinarily consumes
  dense `omega_ssp`. `solvers/irls_direct.py:2730` materializes the coefficient
  pseudoinverse.
- `reml/gradient.py:179` can retain `q` full directional matrices when its
  dense correction branch applies. The compact branch can still retain dense
  products for individual coefficient blocks.
- `reml/penalty_support.py:124`, `:189` and `:234` create wide-precision
  matrices/products. Retained support includes roots, coordinate bases and
  multiple reconstruction, projection and root-error arrays.

Peak memory therefore needs an account of simultaneously live objects:

\[
M_{\mathrm{peak}}=\max_\tau\{
 M_{\mathrm{inputs}}+M_{\mathrm{live\ designs/contexts}}
 +M_{\mathrm{factors}}+M_{\mathrm{precision\ scratch}}
 +M_{\mathrm{probes/recycling}}+M_{\mathrm{certificate}}
 +M_{\mathrm{requested\ outputs}}\}.
\]

Count aliases once, but count independent snapshots and old/new contexts while
both are live. Raw receipts use byte snapshots, which can duplicate substantial
storage. A refinement cache must not retain every discarded hierarchy.
Streaming probes reduces storage, while recycled vectors and retained Lanczos
bases can require `O(p d)` and `O(p ell)` storage. Report retained model bytes
separately from both fit-phase and whole-process peak RSS. Fewer arrays retained
after a fit do not prove a lower high-water mark during construction.

The virtual certificate has an independent cost. For tensor marginal widths
`p_x,p_y` with `P_max = p_x p_y`, the proposed generalized marginal decomposition
conditionally requires `O(p_x^3+p_y^3)` setup,
`O(P_max (p_x+p_y))` per separable inverse action and
`O(P_max+p_x^2+p_y^2)` storage. These expressions require the centered tensor
structure and numerical certificates described in the main design. They omit
the full rich residual scan and nullspace correction, which must be added.
For nullity `k_0`, forming and factoring the nullspace Gram can require
`O(n k_0^2+k_0^3)` work once its evaluation is available. Storing all cross terms
can add `O(P_max k_0)` bytes up to dtype factors; streaming requires further
operator passes. No active-`p`-only memory bound follows from this proposal.

The `HessianFactor` protocol at `solvers/hessian_factor.py:85` and selected
covariance accessors provide existing integration points. An iterative backend
must implement their trace, determinant, selected-block and uncertainty
contracts. Consumers can remain type-specific: `inference/covariance.py:361`
otherwise coerces to a dense array. A coefficient solve alone does not remove
these output and error requirements.

The next experiment is bounded and sequential:

1. Test the context transfer on `rows20` and `rows30` under the existing
   180-second whole-worker deadline. Require preserved rank/certificate
   decisions, numerical agreement, unchanged dispatch, fewer tensor support
   builds and lower complete-fit time without a peak-memory regression.
   Include mutation and zero-face controls. This pilot does not change the
   asymptotic cost of the first support build.
2. For an adaptive fixed-Gaussian prototype, use a five-cell cross:
   `n` in `{2000,20000,200000}` at `p=256`, and `p` in `{64,256,1024}` at
   `n=20000`. Log actual active/virtual widths, overlap, distinct supports,
   closure and fill. Do not manufacture row scaling by repeating identical
   observations. Keep larger coefficient ladders behind the storage audit.
3. Predeclare resource limits, numerical budgets and held-out noninferiority
   before final evaluation. Charge search, certification, failed fits,
   finalization and requested inference. Adaptive acceptance requires lower
   complete-fit time and peak RSS at the declared accuracy. Stop when the
   bound forces essentially full refinement, allocations exceed budget, or
   complete smoothing/inference costs erase the saving.

The numerical and resource limits remain pilot requirements, not claims already
satisfied. Separate profiling from uninstrumented timing; use matched process,
thread and cache conditions and record run variability. A speedup cannot
substitute for the certificate that the method claims to satisfy.

The receipt was independently checked against all three ignored `result.json`
and `run.json` pairs, their saved prediction arrays, and the profiled `fit.prof`.
The audit made 293 comparisons with no discrepancy. It checked 14 artifact
hashes, all 11 recorded profile functions and their caller tuples, runtime and
threadpool identities, source/script/reference identities, timings, counters,
dispatch, fit/REML/EDF/rank values and all 18 response/prediction arrays. Raw
data, transformed-feature and split fingerprints were recomputed from the
recorded parquet input. Saved MSEs and validation/test comparison values were
also recomputed. No fits were rerun.

The audited receipt SHA256 is
`51166cfdec8daa2b52013557d3c7f1f67eeb336397758478dc55f491085d7d81`.

The CPU-model report, pre-run/shared-host narrative and earlier test/environment
check claims are session assertions not independently recoverable from those
run artifacts. They were not treated as verified historical measurements.
