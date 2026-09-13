# Tensor reuse memory measurement plan

Date: 2026-09-13. Benchmark base: `daf6196a6d10b82149eeb146a80363f5c20d5c0b`.
This bounded change implements the retained-memory requirement in
[the cost investigation](2026-09-13-adaptive-interaction-costs.md). It changes
benchmark measurement only. The production reuse change and serial housing
fits belong to the parent task.

Use the existing isolated worktree. Modify only
`benchmarks/benchmark_housing_tensor.py`, `benchmarks/test_housing_tensor.py`,
`benchmarks/housing_tensor.md`, and this note. Keep the helper in the runner so
the existing script fingerprint covers the measurement implementation.

- [x] Add focused failing tests for full NumPy backing owners shared by views,
  independent copies, byte snapshots shared with `np.frombuffer`, memoryviews,
  cyclic containers and model/slot/dataclass objects. Verify that functions,
  classes and modules do not pull their globals into the count. Exercise an
  unsupported external buffer and require an explicit limitation in the result.
- [x] Add `retained_model_storage(model)`. Traverse instance state and ordinary
  containers by identity. Follow array bases and memoryview exporters; charge
  full NumPy owner payloads once and report `bytes` and `bytearray` owner
  payloads separately. Do not serialize the model or claim complete heap/RSS
  coverage. Report unsupported external buffer types instead of treating a
  view's logical size as its allocation.
- [x] Add a synthetic worker regression for measurement order. Sample
  `fit_end_peak_process_rss_mib` immediately after stopping the fit clock,
  before profiler output, telemetry, predictions, export or memory inspection.
  Inspect retained model storage before telemetry and prediction can add state.
  Preserve `peak_process_rss_mib` as the later whole-worker sample through
  prediction export and document that post-fit inspection is now included.
- [x] Document field units, payload exclusions and the versioned retained
  storage scope. Run `uv run --no-sync pytest benchmarks/test_housing_tensor.py
  -q`, `uv run --no-sync ruff check benchmarks/benchmark_housing_tensor.py
  benchmarks/test_housing_tensor.py`, and the corresponding Ruff format check.

The fit-end RSS value is the process high-water through the fit, including
imports, runtime, input data and fitting. It is not a peak of fit-only allocated
bytes. Retained payload counts exclude Python object headers, allocator slack,
native workspaces, released temporaries and buffers the visitor cannot resolve.
Count actual distinct byte snapshots even when their contents match. A bytes
owner used by `np.frombuffer` is charged once in the bytes category.

The parent task must copy this same runner into the baseline checkout and run
matched sequential fits before claiming a time or memory improvement. The
historical RSS receipt cannot supply either of the new measurements.

Verification completed on 2026-09-13. The first test run against the unchanged
runner had nine failures and eight passes. All failures reported the missing
retained-storage measurement. A later weak-reference regression demonstrated
an 8,000-byte overcount through a proxy before the visitor excluded weak
references. The final focused suite passed all 18 tests in 2.27 seconds.
Ruff check, Ruff format check and `git diff --check` passed. The worker tests
use synthetic data and a model double. No housing fits or commits were made
for this measurement change.
