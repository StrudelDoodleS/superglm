# Gaussian preparation: C1 Packet 1

Exact Gaussian chunked fits no longer retain the root's parameter-independent
carrier array. The canonical likelihood ID is computed in blocks of at most
8,192 rows; owned children retain the existing 64 MiB cache allowance and fresh
source checks. Null fitting temporarily binds an eager plan with the same ID.
This removes `8*N` retained root bytes, but does not complete bounded input,
initialization, null fitting or output storage.

The baseline is `141497cfd3117e6d6d4864e90f424babdaa98c16`. The candidate adds
only the six Packet 1 production files. Frozen sources exclude unrelated QR
work. The [receipt](c1_gaussian_preparation_receipt.json) records source, fixture,
driver and raw-artifact hashes, native pools, outputs and separate profiles.

| Complete fit | Baseline wall | Candidate wall | Numerical comparison |
| --- | ---: | ---: | --- |
| Gaussian REML, 1,048,576 rows, q=102 | 20.215 s | 20.729 s | All seven saved arrays bitwise equal |
| Weighted Gaussian, 384 rows | 0.591 s | 0.572 s | Outputs and history exactly equal |
| Weighted Gamma, 384 rows | 0.263 s | 0.270 s | Outputs and history exactly equal |

These are serial fresh-process samples, one pair per fixture. The million-row
public clock includes compilation, optimization and finalization, after an
untimed warmup. BLAS uses one thread, Numba 16 and tabmat OpenMP one; both fits
pass the background-activity screen. Small controls use one native thread and
the existing fixed-penalty fixture with weights from 0.25 to 2.0. Single samples
do not establish a speedup.

The million-row fit retains 8 MiB less root payload. Its measured full-fit peak
RSS is 1,939,943,424 bytes before and 1,955,995,648 after: this pair demonstrates
no whole-fit peak reduction. Both fits converge with rank 102, 16 inner and seven
smoothing iterations. Their backend, complete history, practical-plateau stop,
phase counts and covariance agree exactly. They retain the same 128 MiB of
solver-history rows. The earlier 14.56-second historical timing is faster than
both current samples; the cause has not been isolated. The 12.5-second C1 gate
remains open.

Separate cProfile runs reproduce the timed outputs. Candidate preparation calls
`bind_chunked_likelihood` once from `_fit_candidate` (9.10 ms cumulative),
then `_derived_carrier_digest` once and `_carrier_block` 128 times. The latter
uses 1.70 ms cumulative. Both versions call `_take_likelihood_rows` 320 times:
32 from likelihood evaluation and 288 from derivative chunk iteration. Both
construct 17 eager carriers: the baseline root plus 16 owned children, or the
candidate's temporary null plan plus those children. No extra child replay or
solver iteration appears. Profile clocks are excluded from the timing table.

The affected selection passes **411 tests**. Independent review approves
Packet 1 steps 1–10 without blockers; Ruff and format checks pass on all eight
changed files. The initial regression fails on the unfixed public fit because
37 rows retain 296 carrier bytes. Formula, identity, ownership, mutation,
serialization and fallback regressions cover the new preparation contract.

Raw files and the small-control script are retained under
`.superpowers/sdd/2026-09-11-c1-closeout/`. Million-row measurements reuse
[compact_lss_history.py](compact_lss_history.py) unchanged. Full C1 acceptance
is tracked in the [fixed specification](../docs/superpowers/specs/2026-09-11-c1-closeout.md).
