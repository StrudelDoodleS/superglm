# PSST reference variance: derivation and measurements

PSST now standardizes each ladder rung by its Gaussian-reference variance,
`2 * sum(a**2)`, where `a` contains the shrinkage eigenvalues. Previously it
used `2 * sum(a)`. The two agree on an identified unpenalized block, but
shrinkage makes the old denominator too large. This changes the ranking score;
it does not turn PSST into a calibrated test of a fitted GAM.

The correction covers the dense and structured spline-by-category paths.
The public table keeps its nine columns. Fitting, candidate construction,
dispersion estimation and the EDF budgets keep their existing contracts.

## The reference law

Work on the retained candidate space after projecting out the pair's
intercept and main-effect columns. Let `U` be its working score, `V` its
positive-semidefinite Fisher curvature and `S` its positive-semidefinite
penalty. Fix `lambda >= 0`, assume `A` is positive definite on the retained
space, and write

\[
A=V+\lambda S,\qquad T=U^\top A^{-1}U.
\]

The inverse is restricted to the identified space. A common null direction
of `V` and `S` contributes nothing. Relative to the nuisance-only profiled
optimum, completing the square in

\[
q(b)=U^\top b-\tfrac12b^\top A b
\]

gives `max q = T/2`. This is an exact working-quadratic identity. A complete
refit changes that quadratic and need not achieve the same gain.

Now assume the geometry is fixed and
\(U\sim N(0,\phi V)\), with known dispersion `phi > 0`. Set
\(F=V^{1/2}A^{-1}V^{1/2}\). An orthogonal diagonalization of `F` gives

\[
T/\phi\overset d=\sum_j a_j Z_j^2,\qquad
E(T/\phi)=\sum_j a_j,\qquad
\operatorname{Var}(T/\phi)=2\sum_j a_j^2,
\]

because the independent standard normals have `E[Z²] = 1` and
`Var(Z²) = 2`. Since `A >= V`, every eigenvalue satisfies `0 <= a <= 1`.
Thus `sum(a²) <= sum(a)`, with strict inequality if any identified direction
is partially shrunk. Matching EDF fixes the reference mean, but not the
reference variance.

For example, filters `(0.5, 0.5, 0.5, 0.5)` and `(1, 0.5, 0.25, 0.25)`
both have EDF 2. Their reference variances are respectively 2 and 2.75;
the previous denominator treated both variances as 4. These are exact
fixtures in the regression suite.

The new score is

\[
z_\lambda=
\frac{T_\lambda/\phi-\mathrm{edf}_\lambda}
     {\sqrt{2\sum_j a_j(\lambda)^2}}.
\]

PSST still reports the largest score over the chosen EDF budgets. Those
rungs are dependent. Standardizing each rung does not standardize their
maximum, equalize their tails, or establish the null distribution after
fitting the baseline and estimating weights, dispersion and smoothing.
In particular, normal simulation from the fixed candidate matrix alone
does not supply exact p-values for that entire procedure.

## Dense calculation

The existing generalized factor decomposition gives nonnegative arrays `v`
and `s` such that the filters are `v / (v + lambda*s)`. The correction sums
their squares at the achieved lambda. It reuses the decomposition already
needed for the statistic and EDF, with linear additional work in its width.
Zero-denominator common null directions contribute zero.

The lambda search also uses `sqrt(lo) * sqrt(hi)` for its geometric midpoint.
For positive finite representable endpoints the geometric mean lies between
them. Forming `lo*hi` first can overflow or underflow even when that midpoint
is representable. Extreme penalty-unit regressions reproduced this failure:
the old search returned zero or infinite lambda and missed the requested EDF.

## Structured calculation

Let `G` be the profiled design in its orthogonally compressed row space and
`B` the symmetric retained inverse used by the existing structured statistic
and EDF. Put `V = G.T G` and `F_B = V^(1/2) B V^(1/2)`.
Its row-space smoother is `C = G B G.T`. Cyclicity of trace gives

\[
\operatorname{tr}(F_B^2)=\operatorname{tr}(C^2)=\|C\|_F^2.
\]

When `G (B-A⁺) G.T = 0`, this is the same reference variance derived above.
In floating point the calculation follows
the existing retained-space policy; the consistency checks below do not
establish equivalence to an untruncated original pencil.

The calculation therefore needs squared smoother blocks, including the
off-diagonal blocks between levels. Squaring the EDF or adding only the
diagonal blocks would give the wrong variance.

The existing per-level QR supplies `K_q = R_q T_q⁺`, the cross factor `Y_q`,
and overlap rows `Phi_q`. Put `E_q = K_q Y_q` and `Z_q = [E_q, Phi_q]`.
Using the same resolved, extended and coupled border matrices as the EDF
calculation, the off-diagonal block is

\[
C_{qt}=Z_q M Z_t^\top,\qquad
M=\begin{pmatrix}
\mathrm{resolved}&-\mathrm{extended}\\
-\mathrm{extended}&\mathrm{coupled}
\end{pmatrix}\quad(q\ne t).
\]

Existing PSD factors give each diagonal block as `D_q D_q.T`. The compacted
base block participates in both sums; it cannot be omitted merely because
it has no emitted coefficients. Hence

\[
\|C\|_F^2=
\sum_q\|D_qD_q^\top\|_F^2+
2\sum_{q<t}\|Z_q M Z_t^\top\|_F^2.
\]

In a retained border direction with squared singular value `h`, the raw
two-by-two metric is

\[
\begin{pmatrix}1/h&-1/h\\-1/h&(1-h)/h\end{pmatrix}.
\]

For `h < 1/2`, transform that direction's leaf columns to
`[(E-Phi)/sqrt(h), Phi]`; its metric becomes `diag(1, -1)`. For larger
retained directions the raw metric has norm below 4. An absorbed direction
keeps the projector terms `[[0, -1], [-1, 1-h]]`. Deleting those terms is
incorrect even when the reciprocal is zero. This transformation bounds the
metric; subtraction in the transformed leaf still amplifies error by
`1/sqrt(h)`, which enters the consistency allowance.

A streaming binary QR tree evaluates the cross-level sum. At a merge of
two disjoint leaf sets, their QR factors `R_left` and `R_right` contribute
`2 * ||R_left M R_right.T||²`. Orthogonal transformations preserve this
norm. Every distinct pair of leaves first meets at exactly one merge, so
each off-diagonal block is counted once. The merged rows are then compressed
by QR. All accumulated terms are nonnegative; no large traces are subtracted.

For `L` levels, spline width `k_a` and overlap rank `r`, the local contractions
and tree add at most `O(L*(k_a+r)³)` work: linear in the number of levels at
fixed widths. The tree retains `O(k_a*r + r² log L)` state, including an
uncompressed leaf. Existing chunk buffers bound the other temporaries. No
full cross-level smoother is assembled. Variance is evaluated only at
distinct emitted lambdas, not at every bisection step.
Each final evaluation requires another factor pass and is included in the
caller's evaluation budget. Duplicate budgets share it.

## Numerical scope and regression evidence

For any upper bound `d` on the retained dimension, the reference eigenvalues
imply `EDF²/d <= tr(F²) <= EDF`. The implementation uses the ambient ceiling
`d = L*k_a`, not a separately computed retained rank. It checks these
bounds with an allowance based on its existing PSD assembly model, QR depth,
dimensions, machine epsilon and the transformed leaf's error amplification.
For a smoother perturbation of Frobenius norm at most `delta`, the squared
norm can change by at most `2*||C||_F*delta + delta²`.

These are consistency checks under that assembly model. They are not a
complete forward-error certificate for every operation of the original
factorization or a proof of all rank decisions. Existing rank-gap and
PSD-assembly refusal checks still apply. An invalid final variance refuses
that rung while preserving independent surviving rungs.

Regressions cover the exact equal-EDF examples, unpenalized ranks, zero rank,
penalty-unit rescaling, an actual public smooth-interaction score, and
structured variances against independently assembled observation designs.
The independent oracles factor the augmented design, not the implementation's
block formulas. Tolerances include dimensions, epsilon and conditioning.
Separate tests count actual factor passes and check refusal behavior.

With zero base-level mass and zero penalty, the independent design has four
identified directions, so the smoother is a rank-four projector and its
reference variance is 8. Deleting the absorbed projector terms publishes
5.3333, which the regression rejects. The original `2*EDF` normalization
also failed the penalized fixtures before the correction. An independent
review checked 36 further weighted structured cases against augmented-factor
oracles; the largest norm-scaled variance difference was `3.14e-14`.

## What changed in measured rankings

The 80,000-row freMTPL2 guide example was regenerated, including its two
complete candidate refits. Its top score changes from 1.8455 to 2.3235.
The twelve-pair order is unchanged. The two training deviance gains remain
43.0370 and 73.0481; those gains do not establish held-out benefit.

The null battery was rerun on both revisions: four families, 40 seeds,
8,000 rows per dataset, 160 complete fits and 3,520 candidate pairs per arm.
Neither arm failed a fit or returned a non-finite score. The maximum `ti`
score rises from 7.31 to 9.48 and the maximum `spline_cat` score from 5.53
to 5.66. Unpenalized scores are unchanged. These are measured maxima, not
calibrated rejection thresholds.

A separate small Gaussian experiment used six features and eleven eligible
pairs, with three planted interactions: smooth-by-smooth, numeric-by-category
and category-by-category. The fixed design used seeds 10, 20 and 30 at each
of strengths 0.05, 0.10 and 0.20, with 8,000 training and 8,000 independent
held-out observations per dataset. Every candidate was refitted on both
revisions; all fits converged.

| Signal strength | Planted pairs in top three, across three datasets | Old / corrected |
|---|---|---|
| 0.05 | out of nine | 4 / 4 |
| 0.10 | out of nine | 9 / 9 |
| 0.20 | out of nine | 9 / 9 |

Both revisions selected the same top-three set in all nine datasets, while
the complete order changed in seven. Candidate held-out gains were identical
between revisions. The weak-signal cases include false selections and
negative held-out gains. This small experiment establishes neither a power
improvement nor a universal ranking advantage.

Nor does stronger shrinkage alone guarantee more power. If all `d` identified
directions have the same filter `a > 0`, the factor cancels from the corrected
score: `(a*Q - a*d) / sqrt(2*d*a²) = (Q-d)/sqrt(2*d)`. Equal filters are the
condition; an identity ridge penalty need not give equal filters when Fisher
curvature is anisotropic. Under the unpenalized Gaussian reference, fixing
the dimension and profiled noncentrality also fixes the entire score law,
regardless of how the signal is distributed across whitened directions.

The correction is justified by the reference law. A stronger detector needs
a separate comparison: detection at matched false-positive rates, and
held-out gains for a fixed candidate-refit budget. Correlated main effects,
weak interactions and wider candidate menus are useful next stress cases.
The current evidence supports retaining PSST as a shortlist tool while
building that evaluation, rather than claiming that larger scores solved
interaction selection.

## Complete-fit and screening cost

`benchmarks/screening_reference_variance.py` was run in separate baseline and
corrected processes, sequentially after the other numerical experiments.
Each process warmed compilation, then repeated three complete fits and
screens on 200,000 training rows, with a separate 200,000-row prediction set.
The baseline was `7bbb415c`, whose source tree matches master at `c0ed3a62`.
Python was 3.13.14, NumPy 2.5.2 and SciPy 1.18.0; BLAS and OpenMP used one
thread and Numba used two.

| Case | Fit median, old / corrected | Screen median, old / corrected | Process peak RSS, old / corrected |
|---|---|---|---|
| Six features, eleven dense pairs | 3.720 / 3.417 s | 0.249 / 0.196 s | 502.125 / 501.969 MiB |
| One smooth, 160-level factor | 0.284 / 0.356 s | 0.150 / 0.195 s | 496.566 / 496.762 MiB |

The dense screen ranges overlap: 0.212–0.267 seconds before and 0.181–0.279
after. These three repetitions support no speedup claim. The structured
screen pays for the additional variance pass and tree; its median increases
by about 45 milliseconds in this case. Fit routines were unchanged, and
their timings also vary between runs.

The recorded fit EDF, deviance, dispersion, held-out loss and prediction
summaries are identical across revisions. Every repeated model gives the
same predictions. Dispatch instrumentation observed eleven dense ladder
calls in each mixed screen and one structured call in each wide-factor
screen, with no refusals. Peak RSS includes imports, both datasets, warmup
and repeated models; it is not an isolated screening allocation measurement.
The observed process peaks differ by less than 0.2 MiB.

The reproducible records are
`benchmarks/screening_reference_variance_receipt.json` (performance and all
nine ranking experiments) and
`benchmarks/screening_reference_variance_null_receipt.json` (the paired null
battery). Each performance and ranking run records its source hash. Later
source edits qualify docstrings and bind each tree sibling to a local
variable for static type narrowing; numerical operations and counts are
unchanged. The script records candidate refits
with `--refit`; run its identical file with each revision's environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=2 \
  uv run --no-sync python /path/to/screening_reference_variance.py \
  --case structured --rows 200000 --repeats 3 --output /tmp/receipt.json
```

## Review regression: an affordable binned pair was refused

[Codex identified a routing gap](https://github.com/StrudelDoodleS/superglm/pull/389#discussion_r3997115819)
in the extra-pass accounting. At the default `max_cells=5_000_000`, a
width-11 spline with 5,094 support points and 200 factor levels fits the
allocation gates but has work allowance for only two factor passes. The
penalized ladder needs at least three. The caller admitted the exact support,
received a refusal and returned `NaN` without reaching its binning fallback.

The caller now checks whether two passes can suffice after the allocation
gates, using the built marginal penalty. A nonzero penalty reaches binning;
a zero penalty can still use two passes on the exact support. The marginal
cache avoids building that menu twice. Search and numerical refusals retain
their existing contract.

The public regression failed against `c2be0f6` with a non-finite score before
the routing fix. It now passes, alongside controls that keep the exact
support when the allowance is raised to three or the penalty is zero.
`test_variance_pass_budget_reaches_the_spline_binning_fallback` runs a complete
Gaussian fit and the real structured kernel in all three cases.

`benchmarks/screening_reference_variance_review_receipt.json` records three
complete fits and screens per revision after warmup, using the same
5,094-row dataset and seed 389:

| Measurement | Before routing fix | After routing fix |
|---|---|---|
| Structured support | 5,094 | 256 |
| Work allowance | 2 passes | 680 passes |
| Pair outcome | Refused, `NaN` | `z = 1.128655`, `approx=True` |
| Fit median | 0.098 s | 0.118 s |
| Screen median | 0.248 s | 0.103 s |
| Process peak RSS | 460.715 MiB | 400.496 MiB |

The recorded fit EDF, deviance, dispersion and prediction summaries are
identical. Each screen used one structured ladder call; all three baseline
calls refused and all three corrected calls returned a score. These timings
compare a refusal with a completed binned calculation, so they establish no
general speedup. RSS includes imports, warmup and complete models. Reproduce
the receipt with the benchmark command above, replacing the case with
`--case variance_budget --rows 5094 --seed 389`.
