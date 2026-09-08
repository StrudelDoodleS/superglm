# Recorded C3/C1 performance window

These two scripts are byte-identical snapshots of the controller and summary
used for the final-source benchmark. They retain the original formatting and
two inherited Ruff style diagnostics (UP017 and I001) so the recorded script
hashes and authenticated historical replay remain valid. They are benchmark
evidence, outside the production package.

- `run_timing_final.py` SHA256:
  `7f48497ec3f79d6e50ea0e143e796553f009d808f8a321a2d891d8a446b51f0b`
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
quiet-machine approval. Headroom may remain active and its sampled CPU is
recorded. Review each window independently.

```bash
env -u PYTEST_ADDOPTS uv run python benchmarks/c3_practical/run_timing_final.py \
  --out .benchmark-artifacts/c3-practical/NEW-UNIQUE-WINDOW \
  --max-preflight-external-cores 0.25
uv run python benchmarks/c3_practical/summarize_final_timing_window.py \
  .benchmark-artifacts/c3-practical/NEW-UNIQUE-WINDOW \
  --out /tmp/c3-new-window-summary.json
```

The tracked summary reproduces the existing window's summary exactly when
given its retained raw artifacts. The neutral summary deliberately does not
approve timing claims automatically. See the
[independent receipt](../c3_pragmatic_performance_receipt.json) and
[evidence report](../../docs/research/2026-09-pragmatic-convergence.md) for the
qualified local assessment, numerical comparisons and measurement limitations.
