"""Instrumented call-stack profiling for the Tweedie REML fit path.

Answers, with counters rather than guesses (refs #339 profiling work):

  * how many REML criterion evaluations (``reml_laml_objective``) one fit makes;
  * how many times ``profile_tweedie_reml_scale`` runs per fit;
  * how many criterion evaluations each bounded Brent solve inside it takes
    (``minimize_scalar`` nfev), plus bracket-expansion restarts, curvature
    evaluations, and the phi-cache hit rate;
  * how much wall time the whole scale-profile subsystem accounts for, split
    into the Dunn-Smyth density passes and everything around them.

The probes are import-time monkeypatches on the consumer namespaces (the
callers bind ``profile_tweedie_reml_scale`` by ``from ... import``, so
patching ``superglm.reml.scale`` alone would count nothing).  On a tree whose
``reml.scale`` has no Tweedie profiler (v0.28.0), the scale probes skip and
the harness still counts criterion evaluations, so the same script drives
both arms of an A/B.

Workloads:
  * ``--dataset fremtpl2``: freMTPL2freq (public), y = ClaimNb/Exposure,
    weight = Exposure, s(DrivAge)+s(VehAge)+s(BonusMalus)+Area, Tweedie(1.5).
  * ``--dataset synthetic``: two cr splines + one categorical, CPG-simulated
    Tweedie(1.5) response at ~82% zeros (the shape of the original A/B).
  * ``--dataset random-effect``: RandomEffect(250 levels), n=30k default.

Thread pools are pinned to 1 before numpy import unless
``SUPERGLM_BENCH_NO_PIN`` is set; wall and CPU time are both reported so a
failed pin is visible (CPU >> wall means fan-out).
"""

from __future__ import annotations

import os

if not os.environ.get("SUPERGLM_BENCH_NO_PIN"):
    for _var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ.setdefault(_var, "1")

import argparse
import cProfile
import json
import pstats
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


class Probes:
    """Counters and timers for the REML criterion and its Tweedie scale term."""

    def __init__(self) -> None:
        self.laml_calls = 0
        self.laml_time = 0.0
        self.profile_calls = 0
        self.profile_time = 0.0
        self.minimize_calls = 0
        self.minimize_nfev: list[int] = []
        self.per_profile_minimize: list[int] = []
        self.per_profile_nfev: list[int] = []
        self.sat_calls = 0
        self.sat_cache_hits = 0
        self.sat_time = 0.0
        self.score_calls = 0
        self.score_cache_hits = 0
        self.score_time = 0.0
        self.brentq_calls = 0
        self.brentq_nfev: list[int] = []
        self.density_calls = 0
        self.density_calls_sat = 0
        self.density_calls_score = 0
        self.density_rows = 0
        self.density_time = 0.0
        self.density_time_sat = 0.0
        self.density_time_score = 0.0
        self.prepare_calls = 0
        self.prepare_time = 0.0
        self.pirls_calls = 0
        self.scale_probes_active = False
        self._in_sat = False
        self._in_score = False
        self._in_profile = False
        self._in_laml = 0

    # -- installation -------------------------------------------------------

    def install(self) -> None:
        self._install_laml_probe()
        self._install_scale_probes()

    def _install_laml_probe(self) -> None:
        import importlib

        consumer_names = [
            "superglm.reml.direct",
            "superglm.reml.discrete",
            "superglm.reml.runner",
            "superglm.reml.efs",
            "superglm.reml.scop_efs",
            "superglm.model.reml_finalize",
            "superglm.model.reml_ops",
        ]
        try:
            objective_mod = importlib.import_module("superglm.reml.objective")
        except ImportError:
            return
        orig = getattr(objective_mod, "reml_laml_objective", None)
        if orig is None:
            return
        probes = self

        def counting_laml(*args, **kwargs):
            probes.laml_calls += 1
            probes._in_laml += 1
            t0 = time.perf_counter()
            try:
                return orig(*args, **kwargs)
            finally:
                probes.laml_time += time.perf_counter() - t0
                probes._in_laml -= 1

        for name in consumer_names:
            try:
                mod = importlib.import_module(name)
            except ImportError:
                continue
            if getattr(mod, "reml_laml_objective", None) is orig:
                mod.reml_laml_objective = counting_laml
        objective_mod.reml_laml_objective = counting_laml

    def _install_scale_probes(self) -> None:
        import importlib

        try:
            scale_mod = importlib.import_module("superglm.reml.scale")
        except ImportError:
            return
        orig_profile = getattr(scale_mod, "profile_tweedie_reml_scale", None)
        if orig_profile is None:
            return  # v0.28.0-shaped tree: no Tweedie scale profiler
        self.scale_probes_active = True
        probes = self

        # minimize_scalar is resolved from scale.py's module globals at call
        # time, so patching it there is enough.
        orig_ms = scale_mod.minimize_scalar

        def counting_minimize(*args, **kwargs):
            res = orig_ms(*args, **kwargs)
            probes.minimize_calls += 1
            probes.minimize_nfev.append(int(res.nfev))
            return res

        scale_mod.minimize_scalar = counting_minimize

        # brentq inside scale.py serves both the Gamma profiler and the
        # 0.29.0 Tweedie score polish; count it only inside the Tweedie
        # profile, with the objective wrapped so iteration counts are exact.
        orig_brentq = scale_mod.brentq

        def counting_brentq(f, *args, **kwargs):
            if not probes._in_profile:
                return orig_brentq(f, *args, **kwargs)
            box = [0]

            def counted(u):
                box[0] += 1
                return f(u)

            try:
                return orig_brentq(counted, *args, **kwargs)
            finally:
                probes.brentq_calls += 1
                probes.brentq_nfev.append(box[0])

        scale_mod.brentq = counting_brentq

        def counting_profile(*args, **kwargs):
            calls_before = probes.minimize_calls
            nfev_before = sum(probes.minimize_nfev)
            probes._in_profile = True
            t0 = time.perf_counter()
            try:
                return orig_profile(*args, **kwargs)
            finally:
                probes.profile_time += time.perf_counter() - t0
                probes._in_profile = False
                probes.profile_calls += 1
                probes.per_profile_minimize.append(probes.minimize_calls - calls_before)
                probes.per_profile_nfev.append(sum(probes.minimize_nfev) - nfev_before)

        for name in (
            "superglm.reml.objective",
            "superglm.reml.direct",
            "superglm.reml.discrete",
        ):
            try:
                mod = importlib.import_module(name)
            except ImportError:
                continue
            if getattr(mod, "profile_tweedie_reml_scale", None) is orig_profile:
                mod.profile_tweedie_reml_scale = counting_profile
        scale_mod.profile_tweedie_reml_scale = counting_profile

        orig_prepare = scale_mod.prepare_tweedie_reml_scale_data

        def counting_prepare(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return orig_prepare(*args, **kwargs)
            finally:
                probes.prepare_time += time.perf_counter() - t0
                probes.prepare_calls += 1

        for name in (
            "superglm.reml.objective",
            "superglm.reml.direct",
            "superglm.reml.discrete",
            "superglm.reml.scop_efs",
        ):
            try:
                mod = importlib.import_module(name)
            except ImportError:
                continue
            if getattr(mod, "prepare_tweedie_reml_scale_data", None) is orig_prepare:
                mod.prepare_tweedie_reml_scale_data = counting_prepare
        scale_mod.prepare_tweedie_reml_scale_data = counting_prepare

        orig_sat = scale_mod.TweedieScaleProfileData.saturated_log_likelihood

        def counting_sat(data, phi):
            probes.sat_calls += 1
            if float(phi) in data._saturated_cache:
                probes.sat_cache_hits += 1
            probes._in_sat = True
            t0 = time.perf_counter()
            try:
                return orig_sat(data, phi)
            finally:
                probes.sat_time += time.perf_counter() - t0
                probes._in_sat = False

        scale_mod.TweedieScaleProfileData.saturated_log_likelihood = counting_sat

        orig_score = getattr(scale_mod.TweedieScaleProfileData, "saturated_nll_log_phi_score", None)
        if orig_score is not None:

            def counting_score(data, phi):
                probes.score_calls += 1
                if float(phi) in data._saturated_score_cache:
                    probes.score_cache_hits += 1
                probes._in_score = True
                t0 = time.perf_counter()
                try:
                    return orig_score(data, phi)
                finally:
                    probes.score_time += time.perf_counter() - t0
                    probes._in_score = False

            scale_mod.TweedieScaleProfileData.saturated_nll_log_phi_score = counting_score

        tw = importlib.import_module("superglm.profiling.tweedie")
        orig_eval = tw._evaluate_tweedie_density

        def counting_eval(prepared, phi, **kwargs):
            t0 = time.perf_counter()
            try:
                return orig_eval(prepared, phi, **kwargs)
            finally:
                dt = time.perf_counter() - t0
                probes.density_calls += 1
                probes.density_rows += len(prepared.y)
                probes.density_time += dt
                if probes._in_sat:
                    probes.density_calls_sat += 1
                    probes.density_time_sat += dt
                if probes._in_score:
                    probes.density_calls_score += 1
                    probes.density_time_score += dt

        tw._evaluate_tweedie_density = counting_eval

    # -- reporting ----------------------------------------------------------

    def summary(self) -> dict:
        nfev = np.asarray(self.minimize_nfev, dtype=np.int64)
        per_profile = np.asarray(self.per_profile_nfev, dtype=np.int64)
        return {
            "laml_calls": self.laml_calls,
            "laml_time_s": round(self.laml_time, 4),
            "scale_probes_active": self.scale_probes_active,
            "profile_tweedie_reml_scale_calls": self.profile_calls,
            "profile_time_s": round(self.profile_time, 4),
            "minimize_scalar_calls": self.minimize_calls,
            "bracket_restarts": self.minimize_calls - self.profile_calls,
            "minimize_nfev_total": int(nfev.sum()) if nfev.size else 0,
            "minimize_nfev_per_solve": {
                "min": int(nfev.min()) if nfev.size else 0,
                "median": float(np.median(nfev)) if nfev.size else 0,
                "max": int(nfev.max()) if nfev.size else 0,
            },
            "nfev_per_profile_call": {
                "min": int(per_profile.min()) if per_profile.size else 0,
                "median": float(np.median(per_profile)) if per_profile.size else 0,
                "max": int(per_profile.max()) if per_profile.size else 0,
            },
            "brentq_polish_calls": self.brentq_calls,
            "brentq_polish_nfev_total": int(sum(self.brentq_nfev)),
            "saturated_ll_calls": self.sat_calls,
            "saturated_cache_hits": self.sat_cache_hits,
            "saturated_time_s": round(self.sat_time, 4),
            "score_calls": self.score_calls,
            "score_cache_hits": self.score_cache_hits,
            "score_time_s": round(self.score_time, 4),
            "density_eval_calls": self.density_calls,
            "density_eval_calls_from_value": self.density_calls_sat,
            "density_eval_calls_from_score": self.density_calls_score,
            "density_rows_evaluated": self.density_rows,
            "density_time_s": round(self.density_time, 4),
            "density_time_from_value_s": round(self.density_time_sat, 4),
            "density_time_from_score_s": round(self.density_time_score, 4),
            "prepare_calls": self.prepare_calls,
            "prepare_time_s": round(self.prepare_time, 4),
        }


# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------


def _find_fremtpl2() -> Path:
    for d in (
        os.environ.get("SUPERGLM_DATA_DIR", ""),
        Path.home() / ".cache" / "superglm",
        Path(__file__).resolve().parents[1] / "data",
    ):
        if not d:
            continue
        p = Path(d) / "freMTPL2freq.parquet"
        if p.exists():
            return p
    raise FileNotFoundError("freMTPL2freq.parquet not found in cache/data dirs")


def load_fremtpl2(n_rows: int, seed: int = 42):
    df = pd.read_parquet(_find_fremtpl2())
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
    df["Exposure"] = df["Exposure"].clip(lower=0.01)
    df["DrivAge"] = df["DrivAge"].clip(18, 90)
    df["VehAge"] = df["VehAge"].clip(0, 20)
    df["BonusMalus"] = df["BonusMalus"].clip(50, 150)
    if 0 < n_rows < len(df):
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(df), size=n_rows, replace=False))
        df = df.iloc[idx].reset_index(drop=True)
    y = (df["ClaimNb"] / df["Exposure"]).to_numpy(dtype=float)
    w = df["Exposure"].to_numpy(dtype=float)
    X = df[["DrivAge", "VehAge", "BonusMalus", "Area"]].copy()
    X["Area"] = X["Area"].astype(str)
    return X, y, w


def build_fremtpl2_model(discrete: bool):
    from superglm import Categorical, Spline, SuperGLM, families

    return SuperGLM(
        family=families.tweedie(p=1.5),
        discrete=discrete,
        features={
            "DrivAge": Spline(kind="cr", k=20, penalty="ssp", discrete=discrete),
            "VehAge": Spline(kind="cr", k=15, penalty="ssp", discrete=discrete),
            "BonusMalus": Spline(kind="cr", k=15, penalty="ssp", discrete=discrete),
            "Area": Categorical(),
        },
    )


def load_synthetic(n_rows: int, seed: int = 7, signal: str = "weak"):
    """Two smooth signals + one categorical, CPG Tweedie(1.5), ~82% zeros.

    ``signal="weak"`` puts REML in the heavy-smoothing regime (lambda >> 1,
    flat criterion directions) that the original A/B's 400k case lived in;
    ``signal="strong"`` is the sharply identified regime (lambda < 1).
    """
    from superglm.profiling.tweedie import generate_tweedie_cpg

    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0.0, 1.0, n_rows)
    x2 = rng.uniform(0.0, 1.0, n_rows)
    cat_levels = np.array([f"c{j}" for j in range(8)])
    cat_idx = rng.integers(0, len(cat_levels), n_rows)
    if signal == "strong":
        cat_eff = rng.normal(0.0, 0.3, len(cat_levels))
        eta = 0.8 * np.sin(2.0 * np.pi * x1) + 4.8 * (x2 - 0.5) ** 2 + cat_eff[cat_idx] - 0.5
    else:
        # Nearly penalty-null truths: a faint wiggle on x1 and an exactly
        # linear x2, so REML drives both lambdas into the heavy-smoothing
        # decades where the original A/B's 400k case lived.
        cat_eff = rng.normal(0.0, 0.2, len(cat_levels))
        eta = 0.03 * np.sin(2.0 * np.pi * x1) + 0.25 * (x2 - 0.5) + cat_eff[cat_idx] - 0.5
    mu = np.exp(eta)
    # phi chosen so the zero fraction lands near the 80-85% band of the
    # original A/B: P(y=0) = exp(-mu^(2-p) / (phi (2-p))).
    phi = 9.0
    y = generate_tweedie_cpg(n_rows, mu, phi, 1.5, rng)
    X = pd.DataFrame({"x1": x1, "x2": x2, "cat": cat_levels[cat_idx]})
    w = np.ones(n_rows)
    return X, y, w


def build_synthetic_model(discrete: bool):
    from superglm import Categorical, Spline, SuperGLM, families

    return SuperGLM(
        family=families.tweedie(p=1.5),
        discrete=discrete,
        features={
            "x1": Spline(kind="cr", k=20, penalty="ssp", discrete=discrete),
            "x2": Spline(kind="cr", k=20, penalty="ssp", discrete=discrete),
            "cat": Categorical(),
        },
    )


def load_random_effect(n_rows: int, seed: int = 11):
    from superglm.profiling.tweedie import generate_tweedie_cpg

    rng = np.random.default_rng(seed)
    n_levels = 250
    levels = np.array([f"g{j:03d}" for j in range(n_levels)])
    idx = rng.integers(0, n_levels, n_rows)
    eff = rng.normal(0.0, 0.4, n_levels)
    mu = np.exp(eff[idx] - 0.3)
    y = generate_tweedie_cpg(n_rows, mu, 9.0, 1.5, rng)
    X = pd.DataFrame({"grp": levels[idx]})
    return X, y, np.ones(n_rows)


def build_random_effect_model(discrete: bool):
    from superglm import RandomEffect, SuperGLM, families

    return SuperGLM(
        family=families.tweedie(p=1.5),
        discrete=discrete,
        features={"grp": RandomEffect()},
    )



# --- burn-cost shaped workload (synthetic reproduction of the target shape) ---
#
# 67k rows, Tweedie(1.5) log link, sample weights + offset, five ordered-
# categorical banded axes (13-24 levels) plus four plain categoricals (3-23
# levels), ~83% zero response with gamma-distributed positives.  No real data
# is read: the level universes, the level effects and the exposure are drawn
# from a fixed seed.

_BURN_OC_LEVELS = (13, 17, 20, 24, 15)
_BURN_CAT_LEVELS = (3, 8, 12, 23)
_BURN_INTERCEPT = 5.70
_BURN_PHI = 167.5


def load_burn_cost(n_rows: int, seed: int = 2026):
    from superglm.profiling.tweedie import generate_tweedie_cpg

    if n_rows <= 0:
        n_rows = 67_000
    rng = np.random.default_rng(seed)
    cols: dict[str, np.ndarray] = {}
    eta = np.zeros(n_rows)

    for axis, n_lev in enumerate(_BURN_OC_LEVELS):
        # Banded axes are concentrated in the middle of the range, as real
        # pre-binned pricing axes are.
        raw = rng.normal(0.5, 0.22, n_rows)
        codes = np.clip((raw * n_lev).astype(np.int64), 0, n_lev - 1)
        labels = np.array([f"b{j:02d}" for j in range(n_lev)])
        cols[f"band{axis}"] = labels[codes]
        # A smooth (mostly monotone) effect along the level axis.
        pos = np.linspace(-1.0, 1.0, n_lev)
        shape = (0.55 - 0.14 * axis) * pos + (0.20 - 0.03 * axis) * pos**2
        eta += shape[codes]

    for j, n_lev in enumerate(_BURN_CAT_LEVELS):
        codes = rng.integers(0, n_lev, n_rows)
        labels = np.array([f"f{j}_{k:02d}" for k in range(n_lev)])
        cols[f"fac{j}"] = labels[codes]
        eff = rng.normal(0.0, 0.18, n_lev)
        eff -= eff.mean()
        eta += eff[codes]

    # Exposure-like sample weights and a log-term offset.
    weights = np.clip(rng.gamma(shape=3.0, scale=0.30, size=n_rows), 0.02, 3.0)
    offset = np.log(np.clip(rng.gamma(shape=6.0, scale=2.0, size=n_rows), 1.0, 36.0) / 12.0)

    # Burn cost per unit exposure: currency-scaled, so the Wright argument
    # t = 4 w^2 y / phi^2 lands in the same decades a real book produces.
    intercept = _BURN_INTERCEPT
    mu = np.exp(intercept + eta + offset)
    phi = _BURN_PHI
    y = generate_tweedie_cpg(n_rows, mu, phi / weights, 1.5, rng)
    X = pd.DataFrame(cols)
    return X, y, weights, offset


def build_burn_cost_model(discrete: bool):
    from superglm import Categorical, OrderedCategorical, Spline, SuperGLM, families

    features: dict = {}
    for axis, n_lev in enumerate(_BURN_OC_LEVELS):
        order = [f"b{j:02d}" for j in range(n_lev)]
        features[f"band{axis}"] = OrderedCategorical(
            order=order, basis=Spline(kind="cr", k=min(10, n_lev - 2))
        )
    for j, n_lev in enumerate(_BURN_CAT_LEVELS):
        features[f"fac{j}"] = Categorical()
    return SuperGLM(family=families.tweedie(p=1.5), discrete=discrete, features=features)


WORKLOADS = {
    "fremtpl2": (load_fremtpl2, build_fremtpl2_model),
    "synthetic": (load_synthetic, build_synthetic_model),
    "synthetic-strong": (
        lambda n, seed=7: load_synthetic(n, seed, signal="strong"),
        build_synthetic_model,
    ),
    "random-effect": (load_random_effect, build_random_effect_model),
    "burn-cost": (load_burn_cost, build_burn_cost_model),
}


# ---------------------------------------------------------------------------
# Fit + harvest
# ---------------------------------------------------------------------------


def harvest(model, X, elapsed_wall: float, elapsed_cpu: float, y) -> dict:
    out: dict = {
        "wall_s": round(elapsed_wall, 4),
        "cpu_s": round(elapsed_cpu, 4),
        "n": int(len(y)),
        "zero_fraction": round(float(np.mean(y == 0.0)), 4),
    }
    rr = None
    for attr in ("reml_result_", "reml_result", "_reml_result"):
        rr = getattr(model, attr, None)
        if rr is not None:
            break
    if rr is not None:
        out["lambdas"] = {k: float(v) for k, v in rr.lambdas.items()}
        out["n_reml_iter"] = int(rr.n_reml_iter)
        out["reml_converged"] = bool(rr.converged)
        if rr.objective is not None:
            out["reml_objective"] = float(rr.objective)
    try:
        out["phi_hat"] = float(model.result.phi)
    except Exception:
        pass
    try:
        out["effective_df"] = float(model.result.effective_df)
        out["deviance"] = float(model.result.deviance)
    except Exception:
        pass
    n_probe = min(4096, len(X))
    out["_probe_predictions"] = model.predict(X.iloc[:n_probe])
    return out


def run_once(args) -> dict:
    loader, builder = WORKLOADS[args.dataset]
    loaded = loader(args.rows)
    offset = None
    if len(loaded) == 4:
        X, y, w, offset = loaded
    else:
        X, y, w = loaded

    import superglm

    tree = Path(superglm.__file__).resolve()
    if args.expect_tree and args.expect_tree not in str(tree):
        raise RuntimeError(f"superglm resolved to {tree}, expected *{args.expect_tree}*")

    probes = Probes()
    if not args.no_probes:
        probes.install()

    model = builder(args.discrete)
    prof = cProfile.Profile() if args.cprofile else None
    t_wall = time.perf_counter()
    t_cpu = time.process_time()
    if prof is not None:
        prof.enable()
    model.fit_reml(X, y, sample_weight=w, offset=offset)
    if prof is not None:
        prof.disable()
    elapsed_wall = time.perf_counter() - t_wall
    elapsed_cpu = time.process_time() - t_cpu

    info = harvest(model, X, elapsed_wall, elapsed_cpu, y)
    info["superglm_tree"] = str(tree)
    info["dataset"] = args.dataset
    info["discrete"] = bool(args.discrete)
    info["probes"] = probes.summary() if not args.no_probes else None

    if prof is not None:
        prof.dump_stats(args.cprofile)
        stats = pstats.Stats(prof, stream=sys.stderr)
        stats.sort_stats("cumulative").print_stats(35)

    preds = info.pop("_probe_predictions")
    if args.save_predictions:
        np.save(args.save_predictions, np.asarray(preds, dtype=np.float64))
        info["predictions_file"] = args.save_predictions
    info["predictions_digest"] = {
        "sum": float(np.sum(preds)),
        "min": float(np.min(preds)),
        "max": float(np.max(preds)),
    }
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(WORKLOADS), default="fremtpl2")
    ap.add_argument("--rows", type=int, default=100_000, help="0 means the full dataset")
    ap.add_argument("--discrete", action="store_true")
    ap.add_argument("--no-probes", action="store_true", help="pure timing run, no patches")
    ap.add_argument("--cprofile", type=str, default=None, help="dump .pstats here")
    ap.add_argument("--save-predictions", type=str, default=None, help="save probe preds .npy")
    ap.add_argument("--json", type=str, default=None, help="append result JSON line here")
    ap.add_argument(
        "--expect-tree",
        type=str,
        default=None,
        help="fail unless superglm.__file__ contains this substring",
    )
    args = ap.parse_args()

    info = run_once(args)
    line = json.dumps(info, sort_keys=True)
    print(line)
    if args.json:
        with open(args.json, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


if __name__ == "__main__":
    main()
