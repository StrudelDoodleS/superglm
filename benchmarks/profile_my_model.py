"""Find out where fit() and fit_reml() spend their time on YOUR model.

Edit `load_data()` and `build_model()` at the top, then:

    uv run python benchmarks/profile_my_model.py

Nothing here modifies superglm. It times the public entry points, reads the
timing breakdown superglm already records for REML, and probes the model two
ways the breakdown cannot: which FEATURE costs the most, and how the cost grows
with ROWS.

ALREADY HAVE A FRAME AND A CV ROUTINE?
--------------------------------------
`load_data()` just returns what you already have::

    def load_data():
        return my_X, my_y, my_weights, None

`build_model()` must return ONE **unfitted** model -- this script does the
fitting itself, repeatedly, and a function that fits internally would measure
the wrong thing. If your current function runs k-fold CV, split it in two: the
part that declares `features=` / `family=` moves here, and the fold loop stays
where it is. Typically that means changing::

    def build_and_cv(X, y):                 def make_spec():
        model = SuperGLM(features=..., ...)     return SuperGLM(features=..., ...)
        for train, test in folds: ...       # fold loop stays in your code

then `build_model()` is `return make_spec()`.

Set `N_FOLDS` below to whatever your CV uses. It changes no measurement -- it
only multiplies the single-fit timings out so they can be compared against the
wall time you actually observe around your CV loop. A 5-fold run costs five
fits, which is the usual reason a "slow" model is not actually slow.
"""

from __future__ import annotations

import copy
import cProfile
import io
import pstats
import time
from typing import Any

import numpy as np
import pandas as pd

from superglm import Categorical, OrderedCategorical, Spline, SuperGLM

# ─────────────────────────────────────────────────────────────────────
# EDIT THESE TWO FUNCTIONS (and N_FOLDS if you cross-validate)
# ─────────────────────────────────────────────────────────────────────

# Folds in your CV loop. 1 = you fit once. Only affects the reported
# projection, never a measurement.
N_FOLDS = 1


def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Return (X, y, sample_weight, offset)."""
    rng = np.random.default_rng(0)
    n = 20_000
    levs = [f"L{j:02d}" for j in range(20)]
    X = pd.DataFrame({f"oc{i}": rng.choice(levs, n) for i in range(3)})
    for i in range(6):
        X[f"cat{i}"] = rng.choice([f"C{j}" for j in range(5)], n)
    w = rng.uniform(0.5, 1.0, n)
    y = rng.poisson(0.4 * w).astype(float)
    return X, y, w, None


def build_model() -> SuperGLM:
    """Return an UNFITTED SuperGLM, configured exactly as yours is."""
    levs = [f"L{j:02d}" for j in range(20)]
    feats: dict[str, Any] = {
        f"oc{i}": OrderedCategorical(order=levs, basis=Spline(kind="ps", k=8)) for i in range(3)
    }
    feats.update({f"cat{i}": Categorical() for i in range(6)})
    return SuperGLM(family="poisson", features=feats)


# ─────────────────────────────────────────────────────────────────────
# no need to edit below
# ─────────────────────────────────────────────────────────────────────

RULE = "=" * 72


def _fit(method: str, X, y, w, offset, subset=None, drop=None, profiler=None):
    """Build a fresh model and fit it; return (seconds, model)."""
    model = build_model()
    if drop is not None:
        # Rebuild from the model's own captured constructor contract so the
        # drop-one variant keeps EVERY configured setting (interactions,
        # discretization, penalties, ...), not a cherry-picked subset -- a
        # variant that silently loses configuration times a different model.
        # `_config.constructor_kwargs()` is the one complete record of that
        # contract (it is what `clone_unfitted` reconstructs from).
        cfg = model._config
        kwargs = cfg.constructor_kwargs()
        kwargs["features"] = {k: v for k, v in dict(kwargs["features"]).items() if k != drop}
        interactions = kwargs.get("interactions")
        if interactions:
            kwargs["interactions"] = [
                ia for ia in interactions if not (isinstance(ia, tuple) and drop in ia)
            ]
        targeted = kwargs.get("penalty_features")
        if targeted is not None:
            names = [targeted] if isinstance(targeted, str) else list(targeted)
            kept_targets = [f for f in names if f != drop]
            kwargs["penalty_features"] = kept_targets or None
        model = SuperGLM(**kwargs)

        def _touches_drop(name, spec):
            parents = tuple(getattr(spec, "parent_names", None) or str(name).split(":"))
            return drop in parents

        # constructor_kwargs() carries only pending tuple interactions;
        # explicit interaction OBJECTS live in interaction_templates and are
        # restored by clone_unfitted() after construction. Mirror that here,
        # minus anything touching the dropped feature, so interactions
        # unrelated to the drop keep being part of what gets timed.
        keep = [
            (name, spec)
            for name, spec in cfg.interaction_templates
            if not _touches_drop(name, spec)
        ]
        if keep or cfg.interaction_templates:
            model._interaction_specs = {name: copy.deepcopy(spec) for name, spec in keep}
            kept_names = {name for name, _ in keep}
            model._interaction_order = [n for n in cfg.interaction_order if n in kept_names]
            model._config = type(cfg).capture(model)
    features = model.features
    # With `features=None`/`splines=` configurations the spec dict is empty
    # until fitting auto-detects features; filtering columns against it
    # would time an intercept-only model. Pass the full frame in that case.
    cols = [c for c in X.columns if c in features] if features else list(X.columns)
    Xs, ys = X[cols], y
    ws, os_ = w, offset
    if subset is not None:
        Xs, ys = Xs.iloc[subset], y[subset]
        ws = None if w is None else w[subset]
        os_ = None if offset is None else offset[subset]
    t0 = time.perf_counter()
    # The profiler brackets exactly what the wall clock brackets: the fit
    # call. Enabling it around construction as well would bill build_model()
    # and column selection to a section titled "inside fit_reml()".
    if profiler is not None:
        profiler.enable()
    try:
        getattr(model, method)(Xs, ys, sample_weight=ws, offset=os_)
    finally:
        if profiler is not None:
            profiler.disable()
    return time.perf_counter() - t0, model


def best_of(method, X, y, w, offset, repeat=3, **kw):
    best, model = float("inf"), None
    for _ in range(repeat):
        dt, m = _fit(method, X, y, w, offset, **kw)
        if dt < best:
            best, model = dt, m
    return best, model


def section(title):
    print(f"\n{RULE}\n{title}\n{RULE}")


def main():
    X, y, w, offset = load_data()
    probe = build_model()
    feature_names = list(probe.features)

    section("MODEL")
    print(f"rows        : {len(y):,}")
    print(f"features    : {len(feature_names)}  {feature_names}")
    fam = probe.family
    if isinstance(fam, str):
        fam_desc = fam
    else:
        fam_p = getattr(fam, "p", None)
        fam_desc = type(fam).__name__ + (f"(p={fam_p})" if fam_p is not None else "")
    print(f"family      : {fam_desc}")
    print(f"selection_penalty: {probe.selection_penalty}")

    # ── 1. headline ──
    section("1. HEADLINE  (best of 3 runs each)")
    t_fit, m_fit = best_of("fit", X, y, w, offset)
    print(
        f"fit()       {t_fit:8.3f}s   iters={m_fit.result.n_iter} converged={m_fit.result.converged}"
    )
    try:
        t_reml, m_reml = best_of("fit_reml", X, y, w, offset)
        d = m_reml.reml_diagnostics()
        print(
            f"fit_reml()  {t_reml:8.3f}s   outer={d.get('n_reml_iter')} "
            f"converged={d.get('converged')} termination={d.get('termination_reason')}"
        )
        print(f"\nratio       {t_reml / t_fit:8.1f}x")
        if m_reml._groups:
            print(
                f"coefficients: p={max(g.end for g in m_reml._groups)} across "
                f"{len(m_reml._groups)} groups"
            )
        else:
            print("coefficients: intercept-only (no fitted groups)")
        if N_FOLDS > 1:
            # A fold trains on (k-1)/k of the rows. Whether that costs
            # (k-1)/k of a full fit (row-bound) or nearly a full fit
            # (coefficient/setup-bound) depends on the model, so MEASURE one
            # fold-sized fit per entry point instead of assuming
            # row-linearity. Its own try: a fold subset can be unfittable
            # (a rare required level missing from the sample), and that must
            # not misreport the ALREADY-SUCCEEDED headline fits above.
            try:
                k_rows = max(1, int(len(y) * (N_FOLDS - 1) / N_FOLDS))
                fold_idx = np.random.default_rng(1).choice(len(y), k_rows, replace=False)
                t_fold_fit, _ = best_of("fit", X, y, w, offset, repeat=2, subset=fold_idx)
                print(f"\nprojected over {N_FOLDS} folds -- compare THIS to the time you observe:")
                print(f"  fit()       {t_fold_fit * N_FOLDS:8.3f}s")
                try:
                    t_fold_reml, _ = best_of("fit_reml", X, y, w, offset, repeat=1, subset=fold_idx)
                    print(f"  fit_reml()  {t_fold_reml * N_FOLDS:8.3f}s")
                except Exception:  # noqa: BLE001
                    print("  fit_reml()    failed on the fold-sized subset")
                print(f"  ({N_FOLDS} folds, each measured at its true size of {k_rows:,} rows)")
            except Exception as exc:  # noqa: BLE001
                print(
                    f"\nfold projection skipped: {type(exc).__name__} on the "
                    "fold-sized subset (a rare level missing from the sample?)"
                )
    except Exception as exc:  # noqa: BLE001
        print(f"fit_reml()  FAILED: {type(exc).__name__}: {exc}")
        d, t_reml, m_reml = {}, None, None

    # ── 2. phase breakdown ──
    prof = (d or {}).get("profile") or {}
    if prof:
        section("2. REML PHASE BREAKDOWN  (share of measured total)")
        total = prof.get("total_s") or t_reml or 1.0
        outside = {
            "dm_build_s",
            "fit_runtime_canonicalize_s",
            "fit_prime_caches_s",
            "fit_release_state_s",
        }
        rows = [
            (k, v)
            for k, v in prof.items()
            if k.endswith("_s") and isinstance(v, (int, float)) and k != "total_s"
        ]
        print(f"{'phase':<38}{'seconds':>9}{'share':>8}   where")
        for k, v in sorted(rows, key=lambda kv: -kv[1]):
            where = (
                "before/after optimiser"
                if k in outside
                else ("inside optimiser" if k.startswith("reml_") else "inside IRLS")
            )
            print(f"  {k:<36}{v:8.3f}s{100 * v / total:7.1f}%   {where}")
        inner = (d or {}).get("inner_iter_history") or []
        if inner:
            print(f"\ninner PIRLS iterations: total={sum(inner)}  per outer step={inner}")
        for k in (
            "reml_candidate_reuses",
            "reml_observed_mode_rejected_trial_count",
            "reml_observed_mode_residual_accepted_max",
        ):
            if k in prof:
                print(f"{k}: {prof[k]}")

    # ── 3. cost per feature ──
    section("3. COST PER FEATURE  (refit with each one dropped)")
    print("A large drop means that feature dominates. Uses fit(), which is cheaper.\n")
    print(f"{'dropped':<20}{'fit()':>10}{'saved':>10}{'share':>9}")
    print(f"  {'(nothing)':<18}{t_fit:9.3f}s{'':>10}{'':>9}")
    costs = []
    for name in feature_names:
        try:
            dt, _ = best_of("fit", X, y, w, offset, repeat=2, drop=name)
            saved = t_fit - dt
            costs.append((name, saved))
            print(f"  {str(name):<18}{dt:9.3f}s{saved:9.3f}s{100 * saved / t_fit:8.1f}%")
        except Exception as exc:  # noqa: BLE001
            print(f"  {str(name):<18}  failed: {type(exc).__name__}")
    if costs:
        worst = max(costs, key=lambda kv: kv[1])
        if worst[1] > 0.35 * t_fit:
            print(
                f"\n  >>> {worst[0]!r} alone accounts for {100 * worst[1] / t_fit:.0f}% of fit() time"
            )

    # ── 4. row scaling ──
    section("4. ROW SCALING  (is the cost row-bound or coefficient-bound?)")
    n = len(y)
    rng = np.random.default_rng(0)
    print(f"{'rows':>10}{'fit()':>10}{'fit_reml()':>13}")
    prev = None
    seen_k = set()
    for frac in (0.25, 0.5, 1.0):
        # Cap at the rows that exist: with n < 200 the floor used to ask for
        # 200, silently fit ALL rows at every fraction, and grade a scaling
        # verdict on three identical measurements.
        k = min(n, max(200, int(n * frac)))
        if k in seen_k:
            continue
        seen_k.add(k)
        idx = rng.choice(n, k, replace=False) if k < n else np.arange(n)
        try:
            tf, _ = best_of("fit", X, y, w, offset, repeat=2, subset=idx)
        except Exception as exc:  # noqa: BLE001
            # A random subset can miss a required level (an OC special, a
            # rare category). Report the point and keep the section alive.
            print(f"{k:>10,}   failed: {type(exc).__name__} (subset misses a required level?)")
            continue
        try:
            tr, _ = best_of("fit_reml", X, y, w, offset, repeat=1, subset=idx)
            tr_s = f"{tr:12.3f}s"
        except Exception:  # noqa: BLE001
            tr, tr_s = None, "        failed"
        print(f"{k:>10,}{tf:9.3f}s{tr_s}")
        prev = (k, tf) if prev is None else prev
    if prev:
        k0, t0 = prev
        tf_full, _ = best_of("fit", X, y, w, offset, repeat=2)
        growth = tf_full / max(t0, 1e-9)
        rows_growth = n / max(k0, 1)
        # Three-way: sub-linear growth means fixed/coefficient cost
        # dominates and MORE rows are nearly free -- calling that
        # "row-bound" would reverse the section's answer.
        if growth < 0.5 * rows_growth:
            verdict = "sub-linear: fixed/coefficient cost dominates, rows are cheap"
        elif growth < 1.5 * rows_growth:
            verdict = "row-bound (linear-ish)"
        else:
            verdict = "super-linear, look at coefficient count"
        print(f"\n{rows_growth:.1f}x the rows cost {growth:.1f}x the time -> {verdict}")

    # ── 5. hot functions ──
    section("5. HOT FUNCTIONS INSIDE fit_reml()  (cumulative)")
    pr = cProfile.Profile()
    try:
        _fit("fit_reml", X, y, w, offset, profiler=pr)
    except Exception:  # noqa: BLE001
        _fit("fit", X, y, w, offset, profiler=pr)
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(60)
    shown = 0
    for line in buf.getvalue().splitlines():
        if "superglm" in line and shown < 22:
            print("  " + line.strip())
            shown += 1

    section("HOW TO READ THIS")
    print("""\
reml_linesearch_s dominant      the optimiser is re-fitting IRLS at trial lambdas.
                                Normal; it is the bulk of REML's extra cost.
irls_gram_s dominant            cost is in forming X'WX -- driven by coefficient
                                count (levels x basis size), not rows.
dm_build_s large                design construction, paid once. Large level counts
                                or many features show up here.
fit_runtime_canonicalize_s      post-fit bookkeeping, also paid by plain fit().
outer iterations at the cap     lambda search is not settling -- the real problem.
                                Check termination_reason.
ratio > 4x                      higher than expected; send me this output.""")


if __name__ == "__main__":
    main()
