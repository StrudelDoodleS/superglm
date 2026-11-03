"""Fit cost of shape rules on freMTPL2: one JSON line per configuration.

Poisson, log link, ``log(Exposure)`` offset, claim counts recounted from the
severity file (``examples/fremtpl2_data.py``), a fixed 80/20 split by seed.
Each configuration constrains a different number of spline and categorical
terms; every term is present in every configuration, so the differences isolate
the rules.

    A  three single full-range monotone rules (API of master as well)
    B  A, with ``& concave`` on one of them, a cap on another, and a stretched U
       (``decreasing.below(x) & increasing.above(x)``) on a fourth term
    C  B, with a cap on a fifth spline term and on a sixth, categorical, term

``--configs A`` imports and runs on a master checkout: B and C are built only
when selected.  Threads are not pinned here; pin them on the command line.

    uv run python benchmarks/shape_rules_fit_cost.py --rows 20000
    uv run python benchmarks/shape_rules_fit_cost.py --configs A,B,C --data-dir DIR

Metric sources: wall seconds is ``time.perf_counter`` around the fit call only;
IRLS iterations are the REML profile's ``irls_iters`` (``fit_reml``) or
``result.n_iter`` (``fit``); QP iterations are the summed ``QPResult.n_iter`` of
every ``solve_constrained_qp`` call the fit made (counted by wrapping that
function, which records nothing itself); the backend is
``model.result.direct_backend``; peak RSS is ``ru_maxrss`` of the whole process
(KiB on Linux), so it includes the data load; held-out deviance, NLL, Gini and
A/E come from ``benchmarks/books.glm_results``, the helper every book run uses.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "examples"))

import books  # noqa: E402
import fremtpl2_data  # noqa: E402

import superglm  # noqa: E402
from superglm import Categorical, Constraint, PSpline, SuperGLM  # noqa: E402

SEED = books.SEED
EXPOSURE_FLOOR = 0.01
WARMUP_ROWS = 3000
#: Evaluation spec for ``books.glm_results``: counts, log-exposure offset, no weights.
SPEC = {
    "kind": "glm",
    "response": "claim_count",
    "weights": None,
    "offset": "log_exposure",
    "nll": "poisson_counts",
}


def load_split(data_dir, rows):
    """Train and test frames: the first ``rows`` of a seeded permutation, split 80/20."""
    freq = fremtpl2_data.load_corrected_freq(data_dir)
    freq["Exposure"] = freq["Exposure"].clip(lower=EXPOSURE_FLOOR)
    freq["LogDensity"] = np.log1p(freq["Density"].to_numpy(dtype=np.float64))
    for column in ("DrivAge", "VehAge", "BonusMalus", "VehPower"):
        freq[column] = freq[column].astype(np.float64)
    for column in ("Region", "VehBrand"):
        freq[column] = freq[column].astype(str)
    perm = np.random.default_rng(SEED).permutation(len(freq))
    if rows is not None:
        perm = perm[:rows]
    n_train = int(round(0.8 * len(perm)))
    return (
        freq.iloc[perm[:n_train]].reset_index(drop=True),
        freq.iloc[perm[n_train:]].reset_index(drop=True),
    )


def rules(config):
    """The rule on each constrained term, by name; built only for the selected config."""
    fit = Constraint.fit
    out = {
        "BonusMalus": fit.increasing,
        "LogDensity": fit.increasing,
        "VehAge": fit.decreasing,
    }
    if config in ("B", "C"):
        out["LogDensity"] = fit.increasing & fit.concave
        out["BonusMalus"] = fit.increasing & fit.cap(0.3, 4.0)
        out["DrivAge"] = fit.decreasing.below(55) & fit.increasing.above(55)
    if config == "C":
        out["VehPower"] = fit.cap(0.6, 1.6)
        out["Region"] = fit.cap(0.7, 1.4)
    return out


def build_features(config):
    rule = rules(config)
    return {
        "DrivAge": PSpline(n_knots=8, constraint=rule.get("DrivAge")),
        "VehAge": PSpline(n_knots=12, constraint=rule.get("VehAge")),
        "BonusMalus": PSpline(
            n_knots=12,
            knot_strategy="quantile_tempered",
            knot_alpha=0.2,
            constraint=rule.get("BonusMalus"),
        ),
        "LogDensity": PSpline(n_knots=8, constraint=rule.get("LogDensity")),
        "VehPower": PSpline(n_knots=6, constraint=rule.get("VehPower")),
        "Region": Categorical(constraint=rule.get("Region")),
        "VehBrand": Categorical(),
    }


class QPCount:
    """Sum ``QPResult.n_iter`` over every ``solve_constrained_qp`` call of the fit."""

    def __init__(self):
        self.calls = 0
        self.iterations = 0
        self.module = sys.modules.get("superglm.solvers.irls_direct._qp")
        self.original = getattr(self.module, "solve_constrained_qp", None)

    def __enter__(self):
        if self.original is None:
            return self

        def counted(*args, **kwargs):
            result = self.original(*args, **kwargs)
            self.calls += 1
            self.iterations += int(result.n_iter)
            return result

        self.module.solve_constrained_qp = counted
        return self

    def __exit__(self, *exc):
        if self.original is not None:
            self.module.solve_constrained_qp = self.original


def fit_model(features, train, method, discrete, **opts):
    y = train["ClaimNb"].to_numpy(np.float64)
    offset = np.log(train["Exposure"].to_numpy(np.float64))
    model = SuperGLM(family="poisson", features=features, discrete=discrete)
    started = time.perf_counter()
    getattr(model, method)(train, y, offset=offset, **opts)
    return model, time.perf_counter() - started


def run_config(config, train, test, method, discrete):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warm = train.iloc[:WARMUP_ROWS]
        extra = {"max_reml_iter": 2} if method == "fit_reml" else {}
        fit_model(build_features(config), warm, method, discrete, **extra)  # compile, untimed
        with QPCount() as qp:
            model, wall = fit_model(build_features(config), train, method, discrete)
        held_out = books.glm_results(SPEC, model, test)
    profile = getattr(model, "_reml_profile", None) or {}
    irls = profile.get("irls_iters", getattr(model.result, "n_iter", None))
    rule = rules(config)
    return {
        "config": config,
        "method": method,
        "discrete": discrete,
        "rows": len(train),
        "test_rows": len(test),
        "n_constrained_terms": len(rule),
        "constrained_terms": sorted(rule),
        "wall_s": wall,
        "irls_iterations": irls,
        "qp_calls": qp.calls,
        "qp_iterations": qp.iterations,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "peak_rss_scope": "process-wide, includes the data load",
        "backend": getattr(model.result, "direct_backend", None),
        "converged": bool(getattr(model.result, "converged", False)),
        "deviance": held_out["test_deviance"],
        "nll": held_out["test_nll"],
        "gini": held_out["gini"],
        "ae": held_out["ae_deciles"],
        "ae_worst_dev": held_out["ae_worst_dev"],
        "commit": commit(),
    }


def commit():
    tree = Path(superglm.__file__).resolve().parent
    try:
        out = subprocess.run(
            ["git", "-C", str(tree), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--configs", default="A,B,C", help="comma-separated subset of A,B,C")
    parser.add_argument("--rows", type=int, default=None, help="total rows used (default: all)")
    parser.add_argument("--data-dir", default=None, help="folder with the freMTPL2 parquet files")
    parser.add_argument("--method", default="fit_reml", choices=("fit_reml", "fit"))
    parser.add_argument("--discrete", action="store_true", help="discretised covariates")
    args = parser.parse_args(argv)
    configs = [c.strip().upper() for c in args.configs.split(",") if c.strip()]
    if not set(configs) <= {"A", "B", "C"}:
        parser.error("--configs takes a comma-separated subset of A,B,C")
    data_dir = fremtpl2_data.find_data_dir(args.data_dir)
    train, test = load_split(data_dir, args.rows)
    for config in configs:
        print(json.dumps(run_config(config, train, test, args.method, args.discrete)), flush=True)


if __name__ == "__main__":
    main()
