# Current C1 model baseline at 10 million rows

The existing Gaussian location-scale REML model completes a 10-million-row
fit with `discrete=True` in **152.498 seconds**, using **11.055 GB peak process
RSS**. It converges at rank 102 with 13 inner and five smoothing iterations.
Resident memory immediately after fitting is 6.172 GB; retained solver-history
rows occupy 1.28 GB.

The public fit clock includes predictor compilation (41.31 s), geometry
assembly (76.06 s), optimization and finalization. The complete benchmark
process, including input construction and output work, finishes in 236.84 s
under its 600 s watchdog. These are model-fit measurements, not input-generation
timings or disk-payload sizes.

The [receipt](c1_current_10m_receipt.json) pins the frozen Packet 1 source,
unchanged [driver](compact_lss_history.py), fixture hashes, native pools,
convergence, work counts, activity screen and output hashes. BLAS uses one
thread, Numba 16 and tabmat OpenMP one. The activity screen passes. This is one
sample on the fixed 102-coefficient fragmented Gaussian fixture, not a universal
time or memory guarantee.

The current million-row sample is 20.729 s and 1.956 GB peak process RSS, as
recorded in the [Gaussian preparation report](c1_gaussian_preparation.md).
Iteration counts change with the dataset size, so linear timing extrapolation
is only an estimate.

The user separated 100-million-row/out-of-core fitting into a future roadmap
item. No 100-million-row model has run. The current route still owns full-row
inputs and some live/output state; discretization does not imply constant
whole-fit RAM. The unfinished out-of-core work is preserved locally on
`deferred/c1-out-of-core` at `92f99dda` and is outside this PR's final code.
