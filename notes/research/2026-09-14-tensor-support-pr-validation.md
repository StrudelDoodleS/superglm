# Tensor support handoff review guide

This checkpoint extracts the completed performance change from the larger
interaction research branch. The four production files match the previously
reviewed implementation at `639f499e` byte for byte. The regression module
contains the archived tests plus the nonidentity-coordinate handoff test added
during PR review at `01d2349d`. The research archive retains the original
implementation history; its mutation evidence describes the archived tests.

## Problem and resulting behavior

A discrete Gaussian fit with a fixed tensor penalty built its expensive
selected support during optimization, then built it again during finalization.
The optimizer now returns the populated penalty family. Finalization reuses
the support only after exact construction-input, coordinate, arithmetic and
selected-support evidence matches.

The recipient gets fresh mutable owners and weighted/face state. The obsolete
optimizer owner is released. Unchanged immutable support and its existing
error ledger are shared; changed inputs fall back to ordinary construction.
No smoothing, rank or numerical tolerance policy changes.

## Production review order

1. `src/superglm/reml/penalty_algebra.py`: evidence construction, admission,
   ownership and reuse lifetime.
2. `src/superglm/reml/discrete.py`: optimizer result carries its penalty family.
3. `src/superglm/model/fit_ops.py` and `reml_finalize.py`: select that family,
   admit support transfer and release obsolete owners.
4. `tests/test_penalty_fixed_tensor_reuse.py`: changed-input/arithmetic
   refusals, face ranks, retained ownership and public-fit agreement.

The housing benchmark changes measure fit-end RSS and retained array/buffer
owners through views. That measurement improvement is separate from the
production optimization.

## Recorded complete-fit evidence

These are archived observations from 2026-09-13, not new PR-preparation timing
runs. Both models use 12,384 training observations and one geographic tensor.
Each range contains two observations per implementation on a shared host.

| Housing case | Baseline median, seconds | Candidate median, seconds | Reduction | Baseline / candidate fit-end peak RSS, MiB |
| --- | ---: | ---: | ---: | --- |
| `rows20`, tensor width 361 | 9.38 | 6.19 | 34.0% | 543.57–543.69 / 556.23–556.67 |
| `rows30`, tensor width 841 | 131.23 | 72.37 | 44.8% | 901.18–916.63 / 904.31–904.40 |

All archived prediction arrays and non-timing telemetry match exactly in that
environment. The retained model payload is unchanged. Actual dispatch is the
Gram backend with eight discretized SSP groups and one discretized tensor
group. Profiling counts two expensive tensor support builds before the change
and one afterward. The smaller case has a 2.4% peak-RSS increase; the larger
case's RSS ranges overlap. These observations do not establish a universal
speedup or an asymptotic improvement.

Permanent original-history references:

- [Complete-fit report](https://github.com/StrudelDoodleS/superglm/blob/639f499eaec04967de0b7c31090276671144b99f/docs/research/2026-09-13-tensor-support-handoff-performance.md)
- [Measurements and artifact hashes](https://github.com/StrudelDoodleS/superglm/blob/639f499eaec04967de0b7c31090276671144b99f/docs/research/2026-09-13-tensor-support-handoff-measurements.json)
- [Implementation, numerical contracts and mutation evidence](https://github.com/StrudelDoodleS/superglm/blob/639f499eaec04967de0b7c31090276671144b99f/docs/research/2026-09-13-tensor-support-handoff-report.md)

## Validation boundaries

The original final implementation passed 139 focused tests. Six deliberately
broken admission/face-rank variants were rejected by the regressions. The
original broad run and its worker-environment limitation remain documented in
the complete-fit report. Fresh PR checks are recorded in the PR body and CI,
with their exact head revisions; they do not replace the historical timing
receipts.

The public model API and version/dependency files are unchanged. New compact
representations, adaptive refinement, interaction discovery and additional
Lean contracts are separate research work.
