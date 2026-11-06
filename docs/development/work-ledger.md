# The work ledger

A fit that repeats work, or rebuilds something it could have updated, is slow
in a way that comparing against master cannot show: master has the same waste.
The work ledger lists the expensive operations a fit performs, so the waste is
visible in one table.

```python
from superglm.diagnostics.work_ledger import work_ledger

with work_ledger() as ledger:
    model.fit_reml(X, y)          # a SuperGLM or a SuperLSS fit
print(ledger.report())
```

The ledger is off unless you open the block. While the block is open it swaps a
declared list of functions (`TARGETS` in `superglm/diagnostics/work_ledger.py`)
for wrappers that record each call, and it puts the originals back when the
block ends, even after an error. No production code path changes, so a fit
outside the block costs nothing and gives the same numbers inside the block.

## What it records

For each operation and the place that called it:

- the number of calls, the largest input, and an estimate of the floating-point
  operations (flops);
- the seconds the operation took itself, with the operations it called inside
  it taken out;
- **repeat**: calls whose input is one that operation has already had: the same
  arrays bit for bit, the same numbers and strings, and the same objects (a
  design or a context; the ledger tells objects apart without their memory
  addresses, so a count does not depend on where Python places an object). A
  call that is handed a new settings object each time is never a repeat, and a
  `dict` is compared by its type alone, so a repeat count is a floor;
- **update**: calls whose input differs from the previous call at the same place
  by one row or column (a row and its column, for a symmetric matrix), or by
  one bordering row and column. An update of the earlier result costs about
  n² where a rebuild costs n³.

The operations are the dense factorisations and equilibrations from NumPy,
SciPy and LAPACK, the rank decompositions (`decompose_gram`, `decompose_factor`),
the Gram builds and passes over the rows of a `SuperGLM` fit, and the
likelihood and Gram passes of a `SuperLSS` fit.

## What to expect from the numbers

- The flop figure for a pass over the rows counts it as a dense product, so it
  overstates a binned pass. Read it for ranking, and read the seconds for cost.
- Eigensolver and SVD flops are order-of-magnitude constants.
- Seconds are summed over threads, so a share can pass 100%.
- The ledger hashes every input and keeps one copy of the last input at each
  call site, within a fixed memory budget. This slows a fit while it is on; the
  report says how long the bookkeeping took and leaves it out of the shares.
- An operation the list does not name is not recorded. When the listed
  operations account for little of the fit, the work is somewhere the list does
  not reach: add that entry point to `TARGETS`.
- The equilibrated copy of a system that is solved after being scaled differs
  from the previous one in every entry, so the solve itself is not flagged as an
  update. The equilibration of the unscaled system is.
