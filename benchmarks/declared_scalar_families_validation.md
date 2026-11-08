# SuperLSS declarative families: local validation

LomaxLSS and LogLogisticLSS now share an immutable scalar-family adapter over
the existing symbolic generator. Both support exact responses, left/right
censoring and row truncation through the existing observation contract.
The implementation includes moment-domain behaviour, prediction,
serialization, capability disclosure, warmup coverage and independent
numerical regressions.

The tested source is **6e5639a115e0def2f8b1d213fed3b96eed8eb738** on branch
**t3/fix-superlss-review**, against master baseline
**6643ed9c94497eba0efb8869b945a17c5d380caf**. This evidence and the archived
planning documents are committed afterward; implementation files have not
changed since validation.

Local implementation and verification are complete. Native platform and
remaining supported-version coverage are still open acceptance items.
No push, pull request, merge, version bump, tag or publication occurred.

## Verification

- The main suite reported **22,820 passed, 114 skipped, 43 xfailed**; the
  thread-sensitive phase reported **56 passed, 3 skipped, one non-strict
  xpass**. There were no failures. The runner includes the slow cold-cache
  warmup census.
- All **100 real-data checks** across the five freMTPL2 modules passed with
  SUPERGLM_REQUIRE_DATA=1. None skipped for absent data.
- An isolated Python 3.13 environment with **NumPy 2.3.5 and SciPy 1.15.3**
  passed **126 focused tests**, with three REML/benchmark cases deliberately
  deselected. This covers the tested slice on the SciPy 1.15 line.
- Ruff checks and formatting, generated-source consistency, lock validation,
  installed-dependency checks and run_test.py all exited successfully.

The native environment was Linux x86-64, Python 3.13.14, NumPy 2.5.2,
SciPy 1.18.0 and Numba 0.68.0. Existing warnings, ordinary skips, expected
failures and the non-strict xpass remain visible in the logs. Passing tests
do not certify every stress-fit endpoint.

## Review repairs

All five implementation-review findings now have regression coverage:

1. Family-owned scale-separation disclosure uses the existing censoring
   warn/error/ignore handling and supported-link rules.
2. Stable log-logistic interval values retain tiny endpoint gaps and far-left
   log probabilities. Conditional derivatives vulnerable to cancellation
   are declined on the documented numerical domain.
3. Log-logistic interval jets use endpoint identities, avoiding collapsed
   interior response nodes. Independent checks cover mixed derivatives
   through fourth order, coordinate transformations and normalized likelihoods.
4. Concentrated log-logistic quantiles preserve probability ordering around
   the median.
5. Independent-fit comparisons require stationarity at both endpoints before
   assessing agreement; six intentionally bad endpoint mutations are rejected.

The repair-stage log records 33 failing regressions before those repairs.
The quantile repair separately demonstrated four failures against the unfixed
code. The final full suite exercised the repaired implementation.

## Complete-fit evidence

The [receipt](declared_scalar_families_receipt.json) records source SHAs,
environment, exact commands, per-case timing samples, memory, objectives,
likelihoods, fit status and actual backend. It also hashes the complete local
numerical receipts and logs. Large numerical arrays, generated-work counts and
synthetic held-out metrics remain in those identified local receipts.

The [harness](declared_scalar_families.py) produced **37 dense** and
**12 discrete/compact** fresh-process receipts. Cases include intercepts,
20,000/100,000-row smooth fits, a constant tail, censoring/truncation,
infinite moments, misspecification, GPD overlap and same-session baseline/
current Gamma and log-normal controls.

All 49 receipts reported coefficient convergence without warnings; all 36
smooth fits additionally reported smoothing certification. All 24 paired
comparisons had exactly equal recorded coefficients, parameter predictions,
likelihood/objective, covariance, standard errors, quantiles, EDF, rank and
iterations. Generated work, input and actual backend matched; retained
array/cache differences were zero.

Dense fits dispatched distributional-dense-v1. Requested discrete/compact fits
dispatched distributional-chunked-v1. Eleven untimed instrumentation witnesses
reproduced the timed outputs and dispatch exactly. Profiling was separate
from timing.

Shared/direct pilots use the same generated law and assess adapter overhead.
Independent mathematical checks are in the regression tests. The largest
initial timing gaps received a short warmed within-process ABBA follow-up;
all samples are retained in the receipt. These measurements do not establish
a general speed change. No numerical performance budget was supplied.

Retained-memory subtotals count recognized unique NumPy buffers reachable from
the model, including retained cache arrays. Python object headers and
process-wide native/JIT state are excluded; peak RSS is recorded separately.
Synthetic observed-law scores, exceedance calibration, quantile loss and
bounded-layer quantities exercise the validation workflow. Real-book
tail/layer usefulness still needs an application and held-out data.

## Reproduction

Run from the feature checkout with an isolated baseline checkout:

~~~bash
uv sync --python 3.13 --extra dev
uv run python scripts/fetch_fremtpl.py --dest data/
SUPERGLM_REQUIRE_DATA=1 uv run python scripts/run_test_suite.py -n 4
git worktree add --detach .worktrees/lss-family-baseline 6643ed9c94497eba0efb8869b945a17c5d380caf
uv run python benchmarks/declared_scalar_families.py --baseline-source .worktrees/lss-family-baseline/src --output /tmp/lss-families-dense.json
uv run python benchmarks/declared_scalar_families.py --case lomax-smooth-20k --case log-logistic-smooth-20k --case lomax-smooth-100k --case log-logistic-smooth-100k --case lomax-bounded --case log-logistic-bounded --discrete --compact --output /tmp/lss-families-compact.json
~~~

The measured fetch used the existing pinned local raw cache, via --raw-dir,
after OpenML was unavailable. Both raw-file hashes and the exact measured
commands are in the receipt. Dataset skips were not accepted as data passes.

For a focused numerical/observation reproduction:

~~~bash
uv run pytest tests/test_declared_family_separation.py tests/test_log_logistic_window_stability.py tests/test_lomax_lss_kernel.py tests/test_log_logistic_lss_kernel.py tests/test_lomax_lss_family.py tests/test_log_logistic_lss_family.py tests/test_declared_scalar_adapter.py tests/test_declared_scalar_vertical_slice.py -q
~~~

## Remaining limits

For positive finite log-logistic windows define
r = shape * max(1, abs(log(lo/scale)), abs(log(hi/scale))).
Core derivatives are declined below r = 2^-8; analytic third/fourth derivatives
require r >= 1/8. Order-zero values remain available. This is a numerical
admission restriction for conditional-likelihood cancellation, not a
mathematical shape constraint or uniform error certificate. Derivative
parameters also have the numerical envelope [2^-200, 2^200].
See the [public documentation](../docs/api/distributional/families.md) and
[development notes](../docs/development/internals/distributional-family-development.md).

Two known raw fourth-log-CDF xfails and guarded higher-direction routes remain
disclosed. Separate focused REML stress fits retained four uncertified Lomax
and four uncertified log-logistic endpoints, including iteration-limit stops.
Successful representative benchmarks do not certify those separate endpoints.

Native Windows, macOS and Linux ARM64 and the remaining supported
Python/NumPy/SciPy matrix have not been run. Native focused-family CI coverage
remains necessary before claiming portable acceptance.

The public arbitrary-family factory, general interval censoring,
vector-response Dirichlet fitting, Burr/GB2 and scalar SuperGLM expansion
remain later work.

## Planning and usage

- [Literature-informed roadmap, archived snapshot](../notes/research/2026-11-08-superlss-family-roadmap.md)
- [Concrete framework plan, archived snapshot](../notes/research/2026-11-08-superlss-family-framework-plan.md)
- [Executable Lomax guide](../docs/how-to/fit-a-lomax-distributional-model.md)
