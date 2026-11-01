# Issue 544 stage 2: gamma and generalized-gamma bounds on a declared-derivative framework

2026-11-01, branch `claude/issue-544-gamma-bounds` off master `7aabf3c8`. One PR, opened only
on Max's say-so.

## Goal

`GammaLS` and `GeneralizedGammaLSS` take `truncation=` and `censored=` (the #545 API), with
**analytic natural-parameter derivatives to order 4 on bounded rows**, so REML's LAML endgame
never differences them. The derivative work is built as the C2 framework's core, so later
families (Weibull, Burr XII, log-logistic, Lomax, ...) get bounds and orders 3-4 from a
declaration, with no per-family bounds code.

Inputs, read before building:
- `notes/research/2026-10-31-severity-family-framework.md` (C2 design, prototype, section 3:
  incomplete gamma) and its prototype `notes/research/severity-family-prototype/`;
- `notes/research/2026-10-issue-544-truncated-censored-assessment.md`;
- `/home/max/.cache/superglm-544s2/research/lit_sweep.md` (2026-11-01 sweep: GG near Q = 0 by
  Temme's uniform expansion in the GG's own variables; GST's `d_2` sign; 35 terms for order 4);
- issue #564 (gate error forms for primitives and the incomplete-gamma jets).

## Out of scope (follow-ups, not this PR)

- Public new families (C2 stage 2). Weibull, Burr XII, log-logistic and Lomax declarations
  exist here only as test fixtures for the oracle grid.
- Moving Gaussian, log-normal and GPD bounds onto declarations (would delete their window code
  and give them orders 3-4).
- Replacing hand-written complete-row kernels with generated ones (C2 stage 6).
- Issues #560-#562; interval censoring; the two-piece families.

## Architecture: three layers

1. **Link composition (exists).** `kernels/_curvature_direction.compose_derivatives`,
   `families/_base.natural_curvature_second_direction`.
2. **Bounds, family-free (this PR takes it to order 4).** Window combiner and narrow-window
   quadrature on packed jets; bounded rows' third and fourth derivatives adjusted the way
   `bounded_channels` adjusts orders 0-2.
3. **Per-family derivatives (this PR adds the generator core).** A declaration of log density,
   log survival and log CDF, differentiated by SymPy at build time, emitted as checked-in numba
   kernels; numerical care lives in shared primitives; special-function CDFs enter as
   primitive engines.

## Fixed interfaces

**I1. Packed jets.** For `K` parameters a jet to order `p` is a tuple
`(J0, J1, ..., Jp)`: `J0` shape `(n,)`, `Jk` shape `(n, C(K+k-1, k))`, columns in
`itertools.combinations_with_replacement(range(K), k)` order (`packed_tuples(K, k)`), entries
the plain partial derivatives (not divided by factorials). This matches `packed_pairs`,
`compose_derivatives`' `outer` and every existing third/fourth-derivative array.

**I2. Special-function primitive engines.** A two-argument engine is a numba `@njit` scalar
function `engine(u, v, upper, order, out)` that writes into `out[: 1 + 2 + 3 + ... + (order+1)]`
the value, then the partials in `(u, v)` by total order 1..`order`, each block in
`combinations_with_replacement(range(2), k)` order. `upper` selects the upper tail (survival)
when true, the lower tail (CDF) when false. A row outside the engine's numerical domain writes
NaN everywhere (the kernel marks the row invalid); it never raises. Each engine ships a
vectorised wrapper returning an I1 jet for `K = 2`. Engines:
- `log_gamma_tail(a, x, upper, order, out)`: `log Q(a, x)` (upper) or `log P(a, x)` (lower),
  regularised incomplete gamma, `a` the shape, `x` the scaled point.
- `gg_uniform_log_tail(w, q, upper, order, out)`: `log S` or `log F` of the Prentice
  generalized gamma at standardised `w = (log y - mu)/sigma`, shape `q`, on the uniform
  expansion's domain (`|q| <= 0.2887`, `|q zeta| <= 1`, lit sweep section 1.3), analytic
  through `q = 0`.

**I3. Window jets** (`kernels/_response_tails.py`, new functions beside the order-2 ones, which
stay unchanged): `log_window_jets(near, far, order)` for `log(T(near) - T(far))` and
`quadrature_window_jets(half_width, log_density_jets, order)`, both to order 4. On orders
0-2 they agree with `log_window_channels` and `quadrature_window_channels`.

**I4. Bounded higher derivatives** (`families/_response_bounds.py`):
`bounded_higher_derivatives(third, fourth, *, bounds, multiplier, response, window_jets)`
adjusts complete-row packed third and fourth derivatives exactly as `bounded_channels` adjusts
orders 0-2 (censored rows replaced by `m * window`, truncated rows minus `m * window`). It is
coordinate-agnostic: complete and window jets must be in the same coordinates.

**I5. Generator.** `tools/generate_severity_families.py` (declarations plus SymPy) writes
checked-in numba modules under `src/superglm/distributional/kernels/_declared/`, with a
`--check` mode and a staleness test, the pattern of `tools/generate_lss_fourth_derivatives.py`.
Numba primitive bodies live in `src/superglm/distributional/kernels/_severity_primitives.py`.
SymPy stays a dev dependency; runtime needs only numba and scipy. A declaration can name a
two-argument engine (I2) as a primitive; generated code calls the engine once per row and reads
its partials from `out`.

## Stages

Wave 1 runs in parallel, one fresh agent each in its own worktree. Wave 2 integrates.

| Stage | Owner | Delivers | Gate |
|---|---|---|---|
| S1 window jets | agent | I3 + I4, family-free | SymPy/mpmath oracle on synthetic two-parameter tails, orders 0-4, body and far tails, near-equal ends, one-sided windows; equals the order-2 functions on orders 0-2 |
| S2 generator core | agent | I5, primitives with mpmath tests, gamma (mean, cv) density declaration and prior-weight substitution, oracle grid with Weibull/Burr XII/log-logistic/Lomax as fixtures | #564 stage-1 error forms; mutation check on the `a >= 10` branches; regenerated gamma density orders 0-4 match `tests/test_family_derivatives_symbolic.py` tolerances; `--check` test |
| S3 incomplete-gamma engine | agent | `log_gamma_tail` (I2), series and Legendre continued fraction on order-4 jets in `a`, closed-form `x` partials | mpmath grid `a` in [1e-2, 1e4] across both tails and the diagonal; #564 stage-4a forms (orders 0-2 within 1e-12 for `a <= 12`); orders 3-4 within 1e-7; per-row cost reported |
| S4 GG uniform block | agent | `gg_uniform_log_tail` (I2) by GST's backward recursion in `q^2`, enough terms for order 4 | mpmath grid on the domain, both signs of `q`, through `q = 0`; `dF/dq` at 0 equals `phi(w)(w^2+2)/6`; `d_2` sign settled against mpmath |
| S5 GammaLS bounds | integrator | declared log tails via `log_gamma_tail` (and the uniform block for large `a` near the diagonal if S3's accuracy needs it); #545 checklist; analytic orders 3-4 on bounded plans; separation flags | issue 544's criteria (mean 5,000, CV 1.2 within sampling error); scipy MLE agreement; PIT uniform; tests fail on master; complete fits unchanged |
| S6 GG bounds | integrator | log tails: uniform block inside its domain, `log_gamma_tail` outside, `q` sign switch; location and mean forms; #545 checklist; orders 3-4 | as S5, plus the `q -> 0` window equal to `LogNormalLS`'s to 1e-10 |
| S7 evidence | orchestrator | full suite, complete-fit equality and wall time on all nine families, bounded fit timings, the Burr cell rerun through superglm, docs | AGENTS.md performance evidence; held-out log score through `predict()` |

The #545 checklist for S5 and S6 (from `LogNormalLS`): `censored_response=True` capability;
`bind_observation` in `bind_likelihood`; bounds digest in `plan_identifier`;
`observation.take` in `take`; `censored_carrier`; `validate_likelihood_plan` accepts bounded
plans; `bounded_channels` in `evaluate_natural`; `refuse_bounded` on expected information;
`log_window_probability`; `censored_recession` and `censored_recession_link_sensitive`; the
family names in `model.py`'s refusal message; the import allowlists in
`tests/test_distributional_family_kernel_architecture.py`; the tests that use `GammaLS` as the
family without bounds (`tests/test_lss_response_bounds.py`,
`tests/test_gamma_ls_family.py`).

## Process

- Every pytest, fit, profile or mpmath grid runs through benchboard (`benchboard heavy LABEL --
  timeout N cmd`, `--cores 1` for single-threaded work; plan the label first).
- Agents run their stage's focused tests; the full suite runs once, at S7.
- The quick suite runs before every push. Nothing is pushed and no PR is opened without Max.
- Complete-row behaviour of every family stays bitwise unchanged.
