"""Canonical benchmark for the Tweedie REML power-search cost.

Reproduces the shape of a reported production model: 96,743 rows, 8 features
(6 Categorical, 2 OrderedCategorical with small cr bases), Tweedie/log with an
offset and a wide weight range.

Run before and after any change:

    uv run python benchmarks/tweedie_reml_search_cost.py

The correctness bar is not "it got faster". Any change MUST leave p_hat and
phi_hat unchanged to the tolerances printed at the end -- the search result is
the product, and a faster search that moves the answer is a regression.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from superglm import Categorical, OrderedCategorical, Spline, SuperGLM, families

# Level counts taken from the reported model.
CAT_LEVELS = {"f0": 6, "f1": 9, "f2": 4, "f3": 11, "f4": 13, "f5": 11}
OC_LEVELS = {"f6": 16, "f7": 24}
OC_KNOTS = {"f6": 2, "f7": 1}
N_ROWS = 96_743

# Reference values on this fixture, from origin/master + the parity-skip fix.
# A change that alters these has changed the answer, not just the speed.
REFERENCE = {"p_hat": 1.57461, "phi_hat": 42.05}
P_TOL = 1e-4
PHI_TOL = 5e-2


def build_fixture(n: int = N_ROWS, seed: int = 4):
    rng = np.random.default_rng(seed)
    columns: dict[str, np.ndarray] = {}
    eta = np.full(n, -1.0)
    for name, k in CAT_LEVELS.items():
        levels = [f"{name}_{j:02d}" for j in range(k)]
        idx = rng.integers(0, k, n)
        columns[name] = np.array(levels)[idx]
        eta += rng.normal(0, 0.2, k)[idx]
    oc_levels: dict[str, list[str]] = {}
    for name, k in OC_LEVELS.items():
        levels = [f"{name}_{j:02d}" for j in range(k)]
        idx = rng.integers(0, k, n)
        columns[name] = np.array(levels)[idx]
        eta += 0.02 * (idx - k / 2)
        oc_levels[name] = levels
    frame = pd.DataFrame(columns)
    weights = rng.uniform(1.19e-5, 1.0, n)
    offset = np.where(rng.random(n) < 0.35, 0.0, 1.0986)
    y = np.where(rng.random(n) < 0.83, 0.0, rng.gamma(1.5, np.exp(eta) * 900, n))
    return frame, y, weights, offset, oc_levels


def build_features(oc_levels):
    features = {name: Categorical() for name in CAT_LEVELS}
    for name in OC_LEVELS:
        features[name] = OrderedCategorical(
            order=oc_levels[name],
            basis=Spline(kind="cr", n_knots=OC_KNOTS[name]),
        )
    return features


def _model(oc_levels):
    return SuperGLM(family=families.tweedie(p=1.5), features=build_features(oc_levels))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=N_ROWS)
    parser.add_argument("--json", action="store_true", help="emit machine-readable results")
    args = parser.parse_args()

    frame, y, weights, offset, oc_levels = build_fixture(args.rows)
    results: dict[str, dict] = {}

    def run(label: str, fn):
        t0 = time.perf_counter()
        out = fn()
        elapsed = time.perf_counter() - t0
        results[label] = {"seconds": elapsed, **out}
        return elapsed

    # 1. The slow path under investigation.
    def _reml_search():
        r = _model(oc_levels).estimate_p(
            frame, y, sample_weight=weights, offset=offset, fit_mode="reml"
        )
        return {
            "p_hat": float(r.p_hat),
            "phi_hat": float(r.phi_hat),
            "power_steps": int(r.n_evaluations),
            "phi_evals": int(r.phi_n_evaluations),
            "method": str(r.method),
            "phi_optimizer": str(r.phi_optimizer),
        }

    # 2. The alternative that already exists.
    def _two_step():
        r = _model(oc_levels).estimate_p(frame, y, sample_weight=weights, offset=offset)
        model = SuperGLM(family=families.tweedie(p=r.p_hat), features=build_features(oc_levels))
        model.fit_reml(frame, y, sample_weight=weights, offset=offset)
        return {
            "p_hat": float(r.p_hat),
            "phi_hat": float(r.phi_hat),
            "power_steps": int(r.n_evaluations),
            "phi_evals": int(r.phi_n_evaluations),
        }

    # 3. One REML fit, to show the search cost is the multiplier, not the fit.
    def _single_reml():
        model = _model(oc_levels)
        model.fit_reml(frame, y, sample_weight=weights, offset=offset)
        diag = model.reml_diagnostics()
        return {
            "n_reml_iter": diag.get("n_reml_iter"),
            "profile": {
                k: round(v, 4)
                for k, v in (diag.get("profile") or {}).items()
                if k.endswith("_s") and isinstance(v, (int, float))
            },
        }

    print(f"rows={len(y):,}  features={len(CAT_LEVELS) + len(OC_LEVELS)}\n")
    t_search = run("estimate_p_reml", _reml_search)
    t_two = run("two_step", _two_step)
    t_one = run("single_fit_reml", _single_reml)

    s = results["estimate_p_reml"]
    print(
        f"estimate_p(fit_mode='reml')  {t_search:7.2f}s  steps={s['power_steps']:>2} "
        f"phi_evals={s['phi_evals']:>3}  p={s['p_hat']:.5f} phi={s['phi_hat']:.2f}"
    )
    print(f"two-step (fit -> fit_reml)   {t_two:7.2f}s  p={results['two_step']['p_hat']:.5f}")
    print(
        f"single fit_reml              {t_one:7.2f}s  "
        f"n_reml_iter={results['single_fit_reml']['n_reml_iter']}"
    )
    print(f"\nsearch / two-step ratio      {t_search / t_two:7.2f}x")
    print(f"search / single fit ratio    {t_search / t_one:7.2f}x")

    print("\nper-fit phase breakdown (single fit_reml):")
    for k, v in sorted(results["single_fit_reml"]["profile"].items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {k:<36}{v:8.3f}s")

    dp = abs(s["p_hat"] - REFERENCE["p_hat"])
    dphi = abs(s["phi_hat"] - REFERENCE["phi_hat"])
    ok = dp <= P_TOL and dphi <= PHI_TOL
    print(
        f"\nanswer check vs reference: p delta={dp:.2e} (tol {P_TOL:.0e}), "
        f"phi delta={dphi:.2e} (tol {PHI_TOL:.0e}) -> {'OK' if ok else 'CHANGED -- REGRESSION'}"
    )

    if args.json:
        print("\n" + json.dumps(results, indent=2, default=str))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
