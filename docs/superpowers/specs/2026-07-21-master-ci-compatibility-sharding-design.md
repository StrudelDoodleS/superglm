# Fast Master Compatibility CI Design

## Goal

Keep master as the authoritative Python compatibility gate while reducing its
wall time. Every non-browser test node must still run exactly once on each
supported Python version from 3.10 through 3.14.

## Baseline

The latest successful master `CI` run before this change was GitHub Actions run
`29775889816`. It took 14 minutes 48 seconds. Its five Python jobs ran in
parallel but each took between 10 minutes 41 seconds and 14 minutes 44 seconds.

The workflow currently performs two kinds of redundant work:

- coverage instrumentation runs on all five Python versions although only the
  Python 3.13 report is uploaded;
- Ruff runs inside all five Python jobs and again in the dedicated lint job.

PR #161 separately reduced the dominant wide-Poisson test from 232.61 seconds
to 2.23 seconds without reducing its 50,000-row, 420-column discrete workload.
Its duration-balanced development workflow completes in 1 minute 58 seconds.

## Preserved Test Contract

The new workflow must preserve all of the following:

1. Every test selected by `pytest tests/ -m "not browser"` runs exactly once on
   Python 3.10, 3.11, 3.12, 3.13, and 3.14.
2. No test marker, test fixture, row count, tolerance, or expected failure is
   changed to make CI faster.
3. Python 3.13 continues to produce branch coverage for the complete selected
   suite, not a subset.
4. The four Python 3.13 shard coverage files are combined before XML generation
   and Codecov upload.
5. Browser integration tests remain a single Python 3.13 integration gate.
   Browser repetition across Python versions is not part of the compatibility
   contract because those tests primarily exercise the shared editor/browser
   boundary.
6. Frontend checks and Ruff check/format gates remain independent.
7. `fail-fast: false` remains in test matrices so one version or shard does not
   cancel evidence from the others.
8. The checked-in `.test_durations` file remains the authoritative input for
   duration-balanced partitioning.

## Workflow Architecture

### Compatibility jobs

One matrix job runs the complete non-browser suite without coverage on each of:

- Python 3.10;
- Python 3.11;
- Python 3.12;
- Python 3.14.

These four jobs are unsharded. After removal of the dominant outlier, their
expected test payload is short enough that another fourfold setup fan-out is
not justified.

### Python 3.13 coverage jobs

Four matrix jobs run duration-balanced groups A through D with
`pytest-split`. Their union is the complete non-browser suite on Python 3.13.
Each job writes one coverage data artifact and uses a unique artifact name.

A dependent coverage-combine job downloads all four artifacts, combines the
coverage databases, generates one XML report, and uploads that report to
Codecov. A missing shard artifact or failed combine is a hard failure; Codecov
service availability retains the workflow's existing non-blocking policy.

### Independent checks

The existing dedicated lint job remains the only Ruff invocation. Frontend and
browser jobs remain independent and continue to run in parallel with Python
compatibility jobs.

## Data Flow

```text
3.10 full suite ─┐
3.11 full suite ─┤
3.12 full suite ─┼── authoritative supported-Python evidence
3.14 full suite ─┘

3.13 shard A ─ coverage A ─┐
3.13 shard B ─ coverage B ─┤
3.13 shard C ─ coverage C ─┼── combine ─ coverage.xml ─ Codecov
3.13 shard D ─ coverage D ─┘
```

## Governance and Validation

Repository governance tests will assert:

- the four unsharded compatibility versions;
- all four Python 3.13 groups and the least-duration splitting algorithm;
- absence of coverage flags from compatibility jobs;
- coverage flags and uniquely named artifacts in all coverage shards;
- the coverage-combine dependency and command;
- a single Ruff check and format job;
- retention of both browser test processes.

Local validation will include workflow lint, governance tests, all four shard
commands, representative unsharded supported-Python commands, coverage
combination from four locally isolated data files, Ruff, lock integrity, and
dependency compatibility.

The first post-merge master run is the authoritative hosted performance test.
The target is at most five minutes wall time, with roughly two to four minutes
expected. If the unsharded compatibility jobs exceed that gate after the
wide-Poisson fix, the measured fallback is a four-shard matrix for all Python
versions. That 20-job design is deliberately deferred because it repeats setup
work and increases workflow noise.

## Non-goals

- Do not reduce the supported Python range.
- Do not change production code or numerical behaviour.
- Do not alter test selection or browser coverage.
- Do not make Codecov a required external service.
- Do not shard all five versions without hosted evidence that the simpler
  design misses the five-minute gate.
