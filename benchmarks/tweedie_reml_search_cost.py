"""Canonical benchmark for the Tweedie REML power-search cost.

A synthetic Tweedie/log fixture sized to make the power-search cost visible:
~100k rows, 8 features (6 Categorical, 2 OrderedCategorical with small cr
bases), an offset, and a wide weight range. The dimensions below are chosen to
exercise the search, not to replicate any particular dataset.

Run before and after any change:

    uv run python benchmarks/tweedie_reml_search_cost.py

The correctness bar is not "it got faster". This benchmark fails closed: it
exits non-zero if p_hat or phi_hat drifts past tolerance, and *also* if any
leg reports non-convergence, a non-finite objective, a boundary-pinned
optimum, a density warning, or any Python warning. A search that gives up
early is fast for the wrong reason, and a timing table that cannot tell that
apart from a real speedup is not evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import warnings

import numpy as np
import pandas as pd

from superglm import Categorical, OrderedCategorical, Spline, SuperGLM, families

# Level counts spanning the small-to-moderate range where the per-candidate
# REML fit is cheap enough that the search multiplier dominates the total.
CAT_LEVELS = {"f0": 6, "f1": 9, "f2": 4, "f3": 11, "f4": 13, "f5": 11}
OC_LEVELS = {"f6": 16, "f7": 24}
OC_KNOTS = {"f6": 2, "f7": 1}
N_ROWS = 96_743

# Reference values on this fixture, from origin/master + the parity-skip fix.
# A change that alters these has changed the answer, not just the speed.
# `estimate_p_reml` and `two_step` are different estimators (REML-mode search
# versus ML-mode search), so each carries its own reference rather than being
# held to the other's.
# These constants predate the engine-scoped reml_tol default (1e-6 -> 1e-9 on
# Newton engines) and the always-on publication dispersion re-profile; both
# move phi_hat by up to ~2e-3 relative on this fixture, which PHI_TOL=5e-2
# absorbs. Agreement within tolerance is the contract, not bit-equality with
# the numbers the current code produces.
REFERENCE = {
    "estimate_p_reml": {"p_hat": 1.5746132307, "phi_hat": 42.052},
    "two_step": {"p_hat": 1.5745815576, "phi_hat": 42.054},
    "decoupled_search": {"p_hat": 1.5745815576, "phi_hat": 42.054},
}
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


# ---------------------------------------------------------------------------
# Honesty gates
# ---------------------------------------------------------------------------


def profile_complaints(result) -> list[str]:
    """Name every way a power-search result admits it did not do its job.

    A search that returns unconverged, lands on a configured bound, or profiles
    a non-finite objective is not a cheaper answer to the same question. Each
    flag below is one the search itself sets, so leaving them unchecked means
    the benchmark reports a speedup the search has already disclaimed.
    """
    complaints: list[str] = []
    _missing = object()
    for flag in (
        "converged",
        "outer_converged",
        "fit_converged",
        "solver_converged",
        "objective_finite",
        "phi_converged",
    ):
        value = getattr(result, flag, _missing)
        # Fail closed on a renamed or removed flag: a default of True would
        # read "fine" forever after the field stopped existing.
        if value is _missing:
            complaints.append(f"{flag} missing from the result")
        elif value is False:
            complaints.append(f"{flag}=False")
    # None means "REML did not run here", which is legitimate under fit mode.
    if getattr(result, "reml_converged", None) is False:
        complaints.append("reml_converged=False")
    boundary = getattr(result, "outer_boundary", None)
    if boundary:
        complaints.append(f"outer_boundary={boundary!r} (p_hat pinned to a configured bound)")
    severity = getattr(result, "density_warning_severity", "none")
    if severity not in ("none", "label"):
        complaints.append(f"density_warning_severity={severity!r}")
    if getattr(result, "near_power_boundary", False):
        complaints.append("near_power_boundary=True")
    if getattr(result, "phi_used_fallback", False):
        complaints.append(f"phi_used_fallback=True ({result.phi_fallback_reason})")
    for message in getattr(result, "warnings", None) or []:
        complaints.append(f"result warning: {message}")
    return complaints


def model_complaints(model, label: str) -> list[str]:
    """Name every way a published fit admits it did not converge."""
    complaints: list[str] = []
    result = getattr(model, "result", None)
    if result is not None and getattr(result, "converged", True) is False:
        complaints.append(f"{label}: result.converged=False")
    # Every model reaching this helper is meant to be a published REML fit,
    # so a missing or broken diagnostics payload is itself a complaint --
    # treating it as "not REML" would let a regression that bypasses REML
    # still present successful timings.
    try:
        diagnostics = model.reml_diagnostics()
    except Exception as exc:
        complaints.append(f"{label}: reml_diagnostics unavailable ({type(exc).__name__}: {exc})")
        return complaints
    if diagnostics.get("enabled") is not True:
        complaints.append(f"{label}: reml_diagnostics enabled={diagnostics.get('enabled')!r}")
        return complaints
    if diagnostics.get("converged") is not True:
        complaints.append(f"{label}: reml_diagnostics converged={diagnostics.get('converged')!r}")
    return complaints


def published_backend(model) -> str:
    """The solver backend the published fit actually ran on."""
    solver = model._solver_pirls_result()
    return str(getattr(solver, "direct_backend", None) or "unknown")


class _RssSampler:
    """Sample this process's VmRSS during one leg so each leg gets its OWN
    peak. ru_maxrss is a process-wide high-water mark: after the expensive
    first leg every later leg inherits its value and even a large memory
    regression stays invisible. Sampling misses sub-interval spikes, which
    the printed label discloses; /proc-less platforms fall back to the
    monotone counter with the fallback disclosed the same way."""

    def __init__(self) -> None:
        self.peak_kb = 0.0
        self.sampled = os.path.exists("/proc/self/status")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _vm_rss_kb() -> float:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1])
        return 0.0

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak_kb = max(self.peak_kb, self._vm_rss_kb())
            self._stop.wait(0.002)

    def __enter__(self) -> _RssSampler:
        if self.sampled:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=1.0)
            self.peak_kb = max(self.peak_kb, self._vm_rss_kb())
        else:
            import resource

            self.peak_kb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _run(label: str, fn, results: dict) -> float:
    """Time one leg, capturing any Python warning it raises as a complaint."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with _RssSampler() as sampler:
            start = time.perf_counter()
            payload = fn()
            elapsed = time.perf_counter() - start
    complaints = list(payload.pop("complaints", []))
    complaints += [f"python warning: {w.category.__name__}: {w.message}" for w in caught]
    results[label] = {
        "seconds": elapsed,
        "complaints": complaints,
        "peak_rss_mb": sampler.peak_kb / 1024.0,
        "peak_rss_sampled": sampler.sampled,
        **payload,
    }
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=N_ROWS)
    parser.add_argument("--json", action="store_true", help="emit machine-readable results")
    args = parser.parse_args()

    frame, y, weights, offset, oc_levels = build_fixture(args.rows)
    results: dict[str, dict] = {}

    # 1. The slow path under investigation. The headline leg publishes a
    # fit_reml like the others, so it gets the same published-fit gate:
    # discarding the model here would leave the coupled leg's publication
    # with no convergence check at all.
    def _reml_search():
        model = _model(oc_levels)
        r = model.estimate_p(frame, y, sample_weight=weights, offset=offset, fit_mode="reml")
        return {
            "p_hat": float(r.p_hat),
            "phi_hat": float(r.phi_hat),
            "published_deviance": float(model.result.deviance),
            "backend": published_backend(model),
            "power_steps": int(r.n_evaluations),
            "phi_evals": int(r.phi_n_evaluations),
            "method": str(r.method),
            "phi_optimizer": str(r.phi_optimizer),
            "complaints": profile_complaints(r) + model_complaints(model, "published fit_reml"),
        }

    # 2. The alternative that already exists, spelled out at the call site.
    def _two_step():
        r = _model(oc_levels).estimate_p(frame, y, sample_weight=weights, offset=offset)
        model = SuperGLM(family=families.tweedie(p=r.p_hat), features=build_features(oc_levels))
        model.fit_reml(frame, y, sample_weight=weights, offset=offset)
        return {
            "p_hat": float(r.p_hat),
            "phi_hat": float(r.phi_hat),
            # The timed operation publishes model.fit_reml, so record what it
            # publishes. Its dispersion estimator differs from the profile's
            # (plain fit_reml publishes its solver dispersion, the profile
            # publishes MLE-profiled phi), so equivalence is gated on the
            # estimator-free published deviance instead.
            "published_phi": float(model.result.phi),
            "published_deviance": float(model.result.deviance),
            "backend": published_backend(model),
            "power_steps": int(r.n_evaluations),
            "phi_evals": int(r.phi_n_evaluations),
            "method": str(r.method),
            "complaints": profile_complaints(r) + model_complaints(model, "published fit_reml"),
        }

    # 3. The same trade made inside the library: search under ML, publish REML.
    def _decoupled():
        model = _model(oc_levels)
        r = model.estimate_p(
            frame,
            y,
            sample_weight=weights,
            offset=offset,
            fit_mode="reml",
            search_fit_mode="fit",
        )
        return {
            "p_hat": float(r.p_hat),
            "phi_hat": float(r.phi_hat),
            "published_deviance": float(model.result.deviance),
            "backend": published_backend(model),
            "power_steps": int(r.n_evaluations),
            "phi_evals": int(r.phi_n_evaluations),
            "method": str(r.method),
            "complaints": profile_complaints(r) + model_complaints(model, "published fit_reml"),
        }

    # 4. One REML fit, to show the search cost is the multiplier, not the fit.
    def _single_reml():
        model = _model(oc_levels)
        model.fit_reml(frame, y, sample_weight=weights, offset=offset)
        diagnostics = model.reml_diagnostics()
        return {
            "n_reml_iter": diagnostics.get("n_reml_iter"),
            "backend": published_backend(model),
            "profile": {
                k: round(v, 4)
                for k, v in (diagnostics.get("profile") or {}).items()
                if k.endswith("_s") and isinstance(v, (int, float))
            },
            "complaints": model_complaints(model, "single fit_reml"),
        }

    print(f"rows={len(y):,}  features={len(CAT_LEVELS) + len(OC_LEVELS)}\n")
    t_search = _run("estimate_p_reml", _reml_search, results)
    t_two = _run("two_step", _two_step, results)
    t_decoupled = _run("decoupled_search", _decoupled, results)
    t_one = _run("single_fit_reml", _single_reml, results)

    for label, caption in (
        ("estimate_p_reml", "estimate_p(fit_mode='reml')"),
        ("two_step", "two-step (fit -> fit_reml)"),
        ("decoupled_search", "search_fit_mode='fit'"),
    ):
        row = results[label]
        print(
            f"{caption:<28}{row['seconds']:7.2f}s  steps={row['power_steps']:>2} "
            f"phi_evals={row['phi_evals']:>3}  p={row['p_hat']:.5f} phi={row['phi_hat']:.2f}"
        )
    print(
        f"{'single fit_reml':<28}{t_one:7.2f}s  "
        f"n_reml_iter={results['single_fit_reml']['n_reml_iter']}"
    )
    print(f"\nsearch / two-step ratio      {t_search / t_two:7.2f}x")
    print(f"search / decoupled ratio     {t_search / t_decoupled:7.2f}x")
    print(f"search / single fit ratio    {t_search / t_one:7.2f}x")

    print("\nper-fit phase breakdown (single fit_reml):")
    for k, v in sorted(results["single_fit_reml"]["profile"].items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {k:<36}{v:8.3f}s")

    # Full precision, so a drift smaller than the printed table can still be
    # read off two runs by eye rather than being rounded into agreement.
    print("\nfull-precision estimates:")
    for label in ("estimate_p_reml", "two_step", "decoupled_search"):
        print(
            f"  {label:<20}p_hat={results[label]['p_hat']!r}  phi_hat={results[label]['phi_hat']!r}"
        )

    all_sampled = all(row.get("peak_rss_sampled") for row in results.values())
    rss_caveat = (
        "per-leg sampled maximum; sub-2ms spikes not captured"
        if all_sampled
        else "no /proc: process high-water mark, monotone across legs"
    )
    print(f"\nbackend and memory (peak RSS: {rss_caveat}):")
    for label, row in results.items():
        line = f"  {label:<20}peak_rss={row.get('peak_rss_mb', 0.0):8.1f}MB"
        if row.get("backend"):
            line += f"  backend={row['backend']}"
        print(line)
    backends = {
        label: results[label]["backend"] for label in results if results[label].get("backend")
    }
    if len(set(backends.values())) > 1:
        results["single_fit_reml"]["complaints"].append(
            f"published REML legs ran on different solver backends: {backends}"
        )
    # "unknown" means the solver result lost its dispatch provenance; four
    # matching unknowns would pass an equality-only gate while validating
    # nothing, so unknown itself is a failure.
    for label, backend in backends.items():
        if backend == "unknown":
            results[label]["complaints"].append(
                "published fit lost its solver backend provenance (direct_backend missing)"
            )

    # The two-step leg and the decoupled leg publish the same model up to the
    # search's p resolution, so their published deviances must agree; their
    # dispersions legitimately differ by estimator (solver dispersion vs
    # MLE-profiled), so phi is printed, not gated.
    dev_two = results["two_step"]["published_deviance"]
    dev_dec = results["decoupled_search"]["published_deviance"]
    published_gap = abs(dev_two - dev_dec) / max(abs(dev_dec), 1e-300)
    print(
        f"\npublished two-step vs decoupled: deviance rel gap {published_gap:.2e} "
        f"(gate 1e-4), phi {results['two_step']['published_phi']:.4f} vs "
        f"{results['decoupled_search']['phi_hat']:.4f} (different estimators, not gated)"
    )
    if published_gap > 1e-4:
        results["two_step"]["complaints"].append(
            f"published two-step deviance drifts from the decoupled publication "
            f"by {published_gap:.3e} relative (gate 1e-4)"
        )

    print("\nanswer check vs reference:")
    drifted = False
    if args.rows != N_ROWS:
        print(
            f"  skipped: --rows={args.rows:,} is a different statistical problem than "
            f"the reference constants (calibrated at {N_ROWS:,} rows)"
        )
    else:
        for label, reference in REFERENCE.items():
            dp = abs(results[label]["p_hat"] - reference["p_hat"])
            dphi = abs(results[label]["phi_hat"] - reference["phi_hat"])
            ok = dp <= P_TOL and dphi <= PHI_TOL
            drifted |= not ok
            print(
                f"  {label:<20}p delta={dp:.2e} (tol {P_TOL:.0e})  "
                f"phi delta={dphi:.2e} (tol {PHI_TOL:.0e})  -> {'OK' if ok else 'CHANGED -- REGRESSION'}"
            )

    all_complaints = [(label, c) for label, r in results.items() for c in r["complaints"]]
    print("\nconvergence check:")
    if all_complaints:
        for label, complaint in all_complaints:
            print(f"  {label:<20}{complaint}")
    else:
        print("  all legs converged, no warnings")

    if args.json:
        print("\n" + json.dumps(results, indent=2, default=str))

    if drifted or all_complaints:
        print("\nFAILED: the timings above are not usable evidence.")
    raise SystemExit(1 if (drifted or all_complaints) else 0)


if __name__ == "__main__":
    main()
