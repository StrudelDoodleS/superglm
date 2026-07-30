# Credibility Release-Safety Review Design

**Status:** Approved in conversation on 2026-07-26

**Branch:** `feature/structured-credibility`

**Base review head:** `3c4321c`

## Purpose

Close every justified finding from the independent release-safety review of
PR #165 without redesigning the validated Schur/SZ algebra, touching LSS, or
silently broadening the public deployment contract.

The work remains part of the draft PR. It must finish with behavior-focused
tests, full numerical and release validation, fresh independent and Codex
reviews on the exact final head, and all actionable review threads resolved.
It must not merge or publish the PR.

## Verified Findings

The findings were reproduced against `3c4321c` before this design was written.

| Finding | Verdict | Evidence |
| --- | --- | --- |
| Structured selection accepts unsupported multi-FS topology | Confirmed blocker | Two FS terms and RE-dominant-plus-FS models fit with Gram but fail after `auto` selects structured algebra. |
| Rating-table export silently omits `RandomEffect` | Confirmed blocker | Conditional predictions vary by level while the emitted payload contains no corresponding block and raises no error. |
| Generic term APIs fail with `RandomEffect` | Confirmed | `term_inference`, default `plot`, and `plot_data` raise `TypeError`; `relativities` raises `KeyError`; `drop1` fails only during a reduced fit. |
| Feature and interaction names are not globally unique | Confirmed | A `Numeric` main effect and same-named `FactorSmooth` fit into two coefficient groups; prediction then receives a concatenated vector with the wrong shape. |
| Released reporting state depends on backend | Confirmed | `retain_fit_state=False` reports work after structured fits but fail after forced/automatic Gram fits for RE, FS, and SZ. |
| Discrete FS setup has a large dense transient | Confirmed | At one million rows and `k=10`, process peak RSS rose from about 302 MiB before construction to about 810 MiB. |
| Explicit random-effect reporting rows mix provenance | Confirmed | Evaluation counts and weights appear beside training information, credibility, posterior covariance, and fitted effects under unqualified column names. |
| `splines=` warning is unrelated and undocumented | Partly confirmed | The warning is real, but its removal conflicts with the already approved API decision to deprecate the shorthand. It stays and is documented explicitly. |
| Exact-head CI was not independently visible | Not a defect | The exact `3c4321c` head had 15/15 successful checks and the full local suite had 4,537 passed and 174 skipped. Final-head validation must be repeated. |

## Approved Product Contract

### Rating-table export

Version 0.15.0 does not claim conditional or population-only deployment export
for `RandomEffect`, FS, or SZ terms.

`build_rating_table_payload()` must preflight the whole fitted term topology
before doing impact analysis or emitting any blocks. If an unsupported
structured term is present, it raises one explicit error naming every
unsupported term and stating that no payload was produced.

There is no partial export and no silent population scoring. Conditional
random-effect tables and a receipt-backed population-only mode are separate
future product work.

### Random-effect reporting rows

`random_effects(X=..., y=...)` is a training-reconstruction interface, not an
out-of-time actual-versus-expected interface.

When explicit rows are supplied, their feature geometry, response, weights,
and offset must match the training inputs. A mismatch raises before any report
is constructed. The returned columns therefore retain one coherent training
provenance. Diagnostics record `row_source="fit"`.

Out-of-time diagnostics should later receive a separate API with explicitly
named evaluation quantities.

### Existing shorthand warning

The `splines=` `FutureWarning` remains. Documentation and the PR compatibility
statement must say that:

- existing calls still run in 0.15.0;
- warning-as-error environments will observe a new warning;
- users should migrate to explicit `features={name: Spline(...)}` declarations.

## Structured Backend Eligibility

The structured backend supports at most one dominant factor-smooth block.
Eligibility must enforce that topology before the cost model or matrix
assembly runs.

Selection rules:

1. With no factor smooths, choose the widest eligible `RandomEffect`.
2. With exactly one factor smooth, that term is the only valid dominant
   candidate, even if a random effect is wider.
3. With more than one factor smooth:
   - `direct_solve="auto"` falls back to Gram with an explicit topology reason;
   - `direct_solve="structured"` raises an early ineligibility error naming
     the unsupported terms.
4. Existing constraint, SCOP, local-singularity, and cost-crossover checks
   continue after topology selection.

For an FS-plus-wider-RE model, forced structured solving is valid: the FS block
is dominant and the RE remains in the dense-small partition. Automatic solving
may still choose Gram when that partition makes the cost estimate unfavorable.

No assembly-time unsupported-layout exception should be reachable from a
successful backend decision.

## Safe Generic APIs

### `term_inference`, `plot`, `plot_data`, and `relativities`

`RandomEffect.reconstruct()` will expose its fitted all-level effects through
the conventional `log_relativities` and `relativities` keys while retaining
the existing `effects` mapping.

`term_inference()` will represent a random effect as categorical-shaped
`TermInference` data:

- every fitted level is present;
- the native log relativity is the fitted all-level effect;
- pointwise uncertainty comes from the existing Bayesian coefficient
  covariance;
- native centering is the population-zero penalty target rather than a
  dropped base level;
- display-only mean centering remains available through the existing
  centering option.

The existing categorical plotting paths can then render random effects without
special chart implementations. Covariance helpers must return one standard
error per all-level coefficient, including the inactive fallback shape.

### `drop1`

`drop1()` will reject variance-component models immediately with a precise
message. It will not pretend that the existing fixed-effect likelihood-ratio
procedure is valid for boundary-constrained variance components.

A future REML-aware model-comparison API would need an explicit boundary-test
contract and is out of scope.

### Global names

Public term names must be unique across:

- explicit main features;
- explicit interaction objects;
- generated tuple interactions;
- terms resolved after automatic feature inference.

Validation occurs as soon as both sides of a namespace are known and again
after pending interactions are resolved. The error names both colliding
objects. Generated fitted group names must also remain unique so lambda,
coefficient, and inference dictionaries cannot alias.

## Backend-Neutral Reporting State

Level support is reporting state, not structured-solver state.

The existing immutable RE and factor-smooth support records move behind a
backend-neutral `ReportingSupportState`. Compatibility re-exports from
`superglm.solvers.structured` remain so existing internal imports and tests do
not break during the move.

Finalization becomes:

```text
terminal fitted coefficients and working weights
                    |
                    v
       build backend-neutral support totals
                    |
          +---------+---------+
          |                   |
          v                   v
 structured factors       Gram/QR factors
          |                   |
          +---------+---------+
                    |
                    v
       cache inference, then release rows
```

For every RE/FS/SZ group, support construction records:

- fitted row count;
- fitted analysis weight;
- final local Fisher information;
- training unpooled RE effect when row state will be released.

Structured finalization may reuse its already assembled dominant local
information. Other terms and Gram/QR fits compute the same compact sufficient
statistics from the final working weights before row-scale state is released.

The model owns this reporting state independently of
`StructuredLinearSystemState`. Structured state may reference the same frozen
mapping for compatibility, but reports first consult the backend-neutral
state. Fit-state materialization, transactional projection, coefficient-state
invalidation, deepcopy, and pickle paths all include the new state.

Released-state inference caches continue to own coefficient covariance and
EDF. The new reporting state supplies only level support and does not duplicate
coefficient-sized dense covariance.

## Large-\(n\) Factor-Smooth Construction

### Safe algebra reduction

The current natural-parameterization routine unnecessarily materializes:

- a dense `n x k` basis;
- an `n x k` SVD/rank-check workspace;
- QR `Q`;
- another `n x k` transformed basis.

Only QR `R`, the `k x k` penalty, and row count are needed. The RMS scale is
available analytically:

```text
penalized_scale =
    sqrt(n * rank / sum(1 / positive_generalized_eigenvalues))

null_scale = sqrt(n)
```

The dense compatibility path therefore uses `qr(mode="r")`, checks rank on
the small `R`, and never constructs `Q` or the transformed row basis. A
prototype reduced one-million-row construction from roughly 810 MiB to
604 MiB peak RSS and from 1.62 s to 0.32 s while preserving the same `R` and
predictions.

### Streamed default path

For discrete SZ and the normal FS configuration (`m <= 2` with symmetric
null-component policies), raw basis rows are evaluated in bounded chunks and
combined with tall-skinny QR. Only chunk-local `rows x k` storage and stacked
small `R` factors are live. The final support basis is evaluated only at bin
locations.

The one-million-row prototype reached about 373 MiB process peak RSS
(approximately 71 MiB above the imported-data baseline), completed basis
setup in about 0.18 s, and matched fitted predictions within
`2e-11` on the direct parity case.

The existing QR-whitened eigenparameterization is retained. A replacement
constant/slope null basis was explicitly rejected after it materially degraded
the pinned Poisson the reference implementation cases.

The zero-eigenvalue MRRR coordinates can change by signed permutation under
tall-skinny QR. That is model-equivalent only when the null-component policy
is symmetric. Therefore:

- FS with `m <= 2` and identical null policies uses streaming;
- FS with asymmetric null policies or `m > 2` uses the reduced-memory dense
  compatibility path;
- SZ has no separate null penalties and may always stream;
- tests pin the chosen construction path and protect custom fixed-policy
  predictions.

Documentation and benchmarks state this qualification rather than claiming
that every possible custom FS construction is support-space-only.

No Cython, C/C++, or Rust extension is introduced.

## Test-First Delivery

Every behavior change begins with a failing regression.

### Backend topology

- two FS terms: auto Gram fallback with reason and Gram parity;
- two FS terms: forced structured early rejection;
- one FS plus wider RE: auto fits via the cost decision;
- one FS plus wider RE: forced structured fit and Gram prediction/objective
  parity;
- FS and SZ variants where geometry differs.

### Export and names

- RE, FS, and SZ payload builds fail before producing partial output;
- errors enumerate all unsupported terms;
- main/explicit-interaction and main/generated-interaction collisions fail at
  construction or resolution;
- ordinary unique declarations remain unchanged.

### Public API integration

A simple RE model covers:

- `summary`;
- `term_inference`;
- default and selected `plot_data`;
- default and selected `plot`;
- `relativities`;
- explicit export rejection;
- explicit `drop1` rejection;
- conditional/population prediction;
- deepcopy and pickle.

### Reporting state

For RE, FS, and SZ:

- forced Gram with `retain_fit_state=False`;
- auto selecting Gram with `retain_fit_state=False`;
- forced structured with `retain_fit_state=False`;
- report equality before and after pickle;
- compact-state/no-row-array assertions;
- explicit reporting rows accepted only when they reproduce training geometry,
  response, weights, and offset.

### Large-\(n\) construction

- old and new natural maps, penalty components, fixed-policy fits, and
  predictions agree under the compatible path;
- streaming default FS agrees up to allowed signed null permutation and gives
  prediction/objective parity;
- asymmetric null policies and `m > 2` select compatibility construction;
- pinned the reference implementation FS/SZ cases remain within their existing tolerances;
- an isolated one-million-row benchmark records wall time and peak RSS;
- cProfile/call-stack output confirms basis construction rather than solver
  work owns the measured improvement;
- no time-to-fit regression is accepted.

## Validation and Review Gates

Before any final claim:

1. targeted regression files pass;
2. formatting, Ruff, and mypy pass for touched surfaces;
3. the full local suite passes;
4. package and release verifiers pass;
5. exact/discrete, Gram/structured, FS/SZ, and reference parity gates pass;
6. the isolated one-million-row memory/time benchmark is repeated;
7. an independent code review examines the exact committed head;
8. all actionable findings are fixed and reviewed;
9. Codex is tagged in a fresh PR comment, allowed at least 15 minutes, and its
   exact-head findings are addressed;
10. review comments are resolved only after the corresponding fix is pushed;
11. CI is green on the exact final SHA.

The PR remains draft. Nothing is merged or published.

## Explicit Non-Goals

- No LSS source, test, benchmark, or documentation changes.
- No multi-dominant structured algebra.
- No conditional workbook/SQL export for RE/FS/SZ.
- No implicit population-only deployment export.
- No REML-aware `drop1` implementation.
- No out-of-time random-effect diagnostic API.
- No common REML-driver or Schur/SZ solver redesign.
- No Cython or native-language extension.
