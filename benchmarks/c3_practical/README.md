# Recorded C3/C1 performance window

These scripts derive from the controller and summary used for the final-source
benchmark. The controller now audits external processes without special categories
for obsolete tools. Its current bytes differ from the historical controller.
The summary script is unchanged, but imports the updated controller's audit logic.
Historical receipts retain their original hashes and measurements; their auxiliary
process names and category labels have been generalized. These historical category
totals describe a subset of external processes, not all external activity.

- Historical `run_timing_final.py` SHA256, not the current script:
  `7f48497ec3f79d6e50ea0e143e796553f009d808f8a321a2d891d8a446b51f0b`
- Current `run_timing_final.py` SHA256:
  `93e2deb3d8097be26bcc5df0e60433f5eba7f76295d3f8b0f02a4eb80666faaf`
- `summarize_final_timing_window.py` SHA256:
  `a15df5d8bc52495e4ff0f384ba32dacd287469dce49c1cfc36e7fdf58afa798d`

Run from the implementation worktree with its development environment installed.
The pinned source worktrees must be siblings:

- `c3-c1-baseline`: released source `8962c4520cad948aa20c480a238b7bb1e276e9cd`.
- `c3-pragmatic-final`: final source `5f994c8f6ac0501606594e2f36bfc0cd24050ec1`.
- `c3-c1-completion/data`: the local freMTPL2 data used by the fixture.

Use a new output directory; the controller refuses to reuse one. It launches
six serial workers with numerical threads set to one. Coordinate other
numerical agents and tests before timing. The 0.25-core preflight below
reproduces the second recorded window's screening threshold; it is not
quiet-machine approval. All observed external processes have their sampled CPU
recorded. Review each window independently.

```bash
env -u PYTEST_ADDOPTS uv run python benchmarks/c3_practical/run_timing_final.py \
  --out .benchmark-artifacts/c3-practical/NEW-UNIQUE-WINDOW \
  --max-preflight-external-cores 0.25
uv run python benchmarks/c3_practical/summarize_final_timing_window.py \
  .benchmark-artifacts/c3-practical/NEW-UNIQUE-WINDOW \
  --out /tmp/c3-new-window-summary.json
```

The current summary requires a controller hash matching the current script, so
historical replay requires the historical controller from repository history.
New runs use the updated external-process categories. The summary does not
approve timing claims automatically. See the
[independent receipt](../c3_pragmatic_performance_receipt.json) and
[evidence report](../../docs/research/2026-09-pragmatic-convergence.md) for the
qualified local assessment, numerical comparisons and measurement limitations.
