"""Paired PSST normalization study; see psst_detection_protocol.md.

Example: uv run python benchmarks/psst_detection_study.py --phase pilot
Raw resumable records are written outside tracked production paths.
"""

from __future__ import annotations

# Thread limits must precede NumPy/SciPy imports in fresh worker processes.
# ruff: noqa: E402
import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import resource
import subprocess
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np
import pandas as pd
import scipy

import superglm
import superglm.model.screening_ops as ops
from superglm import SuperGLM
from superglm.features import Categorical, Spline

CASES = {
    "gaussian_bilinear": ("gaussian", "bilinear", False),
    "gaussian_wave": ("gaussian", "wave", False),
    "gaussian_spline_cat": ("gaussian", "spline_cat", False),
    "gaussian_cat_cat": ("gaussian", "cat_cat", False),
    "poisson_wave": ("poisson", "wave", False),
    "gaussian_correlated_wave": ("gaussian", "wave", True),
}
NULL_CASES = ("gaussian_wave", "poisson_wave", "gaussian_correlated_wave")
STRENGTHS = (0.02, 0.04, 0.06, 0.08, 0.12)
SUPPORT = np.linspace(-1.0, 1.0, 41)
SINE_SCALE = float(np.sqrt(np.mean(np.sin(np.pi * SUPPORT) ** 2)))
LINEAR_SCALE = float(np.sqrt(np.mean(SUPPORT**2)))


def rank_ladders(pairs, ladders, phi):
    """Score all rungs independently for each denominator, retaining identity."""
    if len(pairs) != len(ladders) or not phi > 0:
        raise ValueError("One complete ladder per candidate and positive dispersion required")
    ranked = {"old": [], "corrected": []}
    for pair, ladder in zip(pairs, ladders, strict=True):
        for method in ranked:
            choices = []
            for rung, result in enumerate(ladder):
                variance = 2 * result.edf0 if method == "old" else result.reference_variance
                if result.edf0 > 0 and variance > 0:
                    z = (result.statistic / phi - result.edf0) / math.sqrt(variance)
                    if math.isfinite(z):
                        choices.append((z, -rung, result))
            if not choices:
                raise ValueError(f"No finite scoring rung for {pair}")
            z, neg_rung, result = max(choices, key=lambda item: item[:2])
            ranked[method].append(
                {
                    "pair": ":".join(pair),
                    "z": float(z),
                    "rung": -neg_rung,
                    "edf": float(result.edf0),
                    "lambda": float(result.lambda0),
                }
            )
    for rows in ranked.values():
        rows.sort(key=lambda row: (-row["z"], row["pair"]))
    return ranked


def select_by_validation(shortlist, results):
    choices = ["baseline", *(pair for pair in shortlist if "validation_loss" in results[pair])]
    return min(choices, key=lambda pair: results[pair]["validation_loss"])


def design_key(case):
    family, _, correlated = CASES[case]
    return family + ("_correlated" if correlated else "_independent")


def features():
    return {
        **{f"s{i:02}": Spline(kind="ps", k=(6, 8, 10)[i % 3]) for i in range(12)},
        **{f"c{i:02}": Categorical() for i in range(18)},
    }


def sample(case, strength, replicate, phase, stream, n):
    family, shape, correlated = CASES[case]
    design = {"gaussian_independent": 0, "poisson_independent": 1, "gaussian_correlated": 2}
    phase_index = {"pilot": 0, "calibration": 1, "audit": 2, "signal": 3}[phase]
    seed = [20260913, design[design_key(case)], phase_index, replicate, stream]
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    smooth = SUPPORT[rng.integers(0, len(SUPPORT), size=(n, 12))]
    if correlated:
        for source, target in ((1, 3), (2, 4)):
            copy_mask = rng.random(n) < 0.7
            smooth[copy_mask, target] = smooth[copy_mask, source]
    cats, contrasts = {}, {}
    for i in range(18):
        levels = (2, 3, 5)[i % 3]
        values = rng.integers(0, levels, size=n)
        cats[f"c{i:02}"] = values.astype(str)
        contrasts[i] = (values - (levels - 1) / 2) / math.sqrt((levels**2 - 1) / 12)
    frame = pd.DataFrame({**{f"s{i:02}": smooth[:, i] for i in range(12)}, **cats})
    additive = 0.1 * np.sin(np.pi * smooth).sum(axis=1)
    additive += 0.06 * sum(contrasts.values())
    sine = np.sin(np.pi * smooth[:, 1]) / SINE_SCALE
    if shape == "bilinear":
        signal = smooth[:, 1] * smooth[:, 2] / LINEAR_SCALE**2
        target = "s01:s02"
    elif shape == "wave":
        signal = sine * np.sin(np.pi * smooth[:, 2]) / SINE_SCALE
        target = "s01:s02"
    elif shape == "spline_cat":
        signal = sine * contrasts[1]
        target = "s01:c01"
    else:
        signal = contrasts[1] * contrasts[2]
        target = "c01:c02"
    eta = additive + strength * signal
    mean = eta if family == "gaussian" else np.exp(0.5 + eta)
    response = (
        mean + rng.normal(size=n) if family == "gaussian" else rng.poisson(mean).astype(float)
    )
    return frame, response, mean, target


def fit(frame, y, family, interactions=()):
    model = SuperGLM(family=family, features=features(), interactions=list(interactions)).fit_reml(
        frame, y
    )
    diagnostics = model.reml_diagnostics()
    if not model.result.converged or (diagnostics["enabled"] and not diagnostics.get("converged")):
        raise RuntimeError(f"Fit did not converge: {diagnostics}")
    return model


@contextmanager
def capture_ladders(legacy=False):
    original = ops.penalized_score_statistic_ladder
    original_structured = ops.structured_ladder
    captured = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result)
        return [replace(r, reference_variance=2 * r.edf0) for r in result] if legacy else result

    def unexpected_structured(*args, **kwargs):
        raise RuntimeError("Protocol requires exact dense dispatch; structured path was called")

    ops.penalized_score_statistic_ladder = capture
    ops.structured_ladder = unexpected_structured
    try:
        yield captured
    finally:
        ops.penalized_score_statistic_ladder = original
        ops.structured_ladder = original_structured


def screen(model, frame, y, verify_old=False):
    pairs = list(itertools.combinations(frame.columns, 2))
    with capture_ladders() as ladders:
        table = model.screen_interactions(frame, y, candidates=pairs)
    if len(table) != 435 or table["approx"].any() or not np.isfinite(table["z"]).all():
        raise RuntimeError("Expected 435 finite, exact candidate scores")
    ranked = rank_ladders(pairs, ladders, table.attrs["phi"])
    public = {f"{r.feature_a}:{r.feature_b}": r.z for r in table.itertuples()}
    for row in ranked["corrected"]:
        if row["z"] != public[row["pair"]]:
            raise RuntimeError("Captured corrected score differs from public screen")
    if verify_old:
        with capture_ladders(legacy=True):
            old = model.screen_interactions(frame, y, candidates=pairs)
        public_old = {f"{r.feature_a}:{r.feature_b}": r.z for r in old.itertuples()}
        for row in ranked["old"]:
            if row["z"] != public_old[row["pair"]]:
                raise RuntimeError("Captured old score differs from old-denominator public replay")
    return ranked, {
        "dense_ladders": len(ladders),
        "structured_ladders": 0,
        "phi": table.attrs["phi"],
    }


def loss(y, prediction, family):
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Loss requires finite outcomes and predictions")
    if family == "gaussian":
        value = float(np.mean((y - prediction) ** 2))
    else:
        if np.any(prediction <= 0):
            raise ValueError("Nonpositive Poisson prediction")
        # The outcome-only log-factorial cancels in model comparisons.
        value = float(np.mean(prediction - y * np.log(prediction)))
    if not math.isfinite(value):
        raise ValueError("Computed loss must be finite")
    return value


def prediction_risk(mean, prediction, family):
    if not np.isfinite(mean).all() or not np.isfinite(prediction).all():
        raise ValueError("Risk requires finite means and predictions")
    if family == "gaussian":
        value = float(np.mean((mean - prediction) ** 2))
    else:
        if np.any(mean <= 0) or np.any(prediction <= 0):
            raise ValueError("Nonpositive Poisson mean or prediction")
        value = float(np.mean(prediction - mean + mean * np.log(mean / prediction)))
    if not math.isfinite(value):
        raise ValueError("Computed risk must be finite")
    return value


def score_frozen_selections(winners, fitted_models, test, y, mean, family):
    """Test failures are evaluation failures; they never cause reselection."""
    measured = {}
    for pair in sorted({"baseline", *winners.values()}):
        try:
            prediction = fitted_models[pair].predict(test)
            measured[pair] = {
                "test_loss": loss(y, prediction, family),
                "test_risk": prediction_risk(mean, prediction, family),
            }
        except Exception as error:
            measured[pair] = {"test_error": f"{type(error).__name__}: {error}"}
    outcomes = {}
    for method, winner in winners.items():
        outcomes[method] = {"selected": winner}
        failures = [
            measured[p]["test_error"] for p in ("baseline", winner) if "test_error" in measured[p]
        ]
        if failures:
            outcomes[method]["evaluation_error"] = failures
        else:
            for metric in ("test_loss", "test_risk"):
                outcomes[method][metric + "_gain"] = (
                    measured["baseline"][metric] - measured[winner][metric]
                )
    return outcomes, measured


def run_trial(task):
    started = time.perf_counter()
    case, strength, replicate, phase, do_refit = task
    record = {
        "id": f"{phase}/{case}/{strength:g}/{replicate}",
        "case": case,
        "design": design_key(case),
        "strength": strength,
        "replicate": replicate,
        "phase": phase,
        "status": "failed",
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            family = CASES[case][0]
            train, y, _, target = sample(case, strength, replicate, phase, 0, 4000)
            tick = time.perf_counter()
            model = fit(train, y, family)
            record["fit_seconds"] = time.perf_counter() - tick
            tick = time.perf_counter()
            ranked, dispatch = screen(model, train, y, verify_old=phase == "pilot")
            record.update(
                screen_seconds=time.perf_counter() - tick, dispatch=dispatch, target=target
            )
            record["methods"] = {}
            for method, rows in ranked.items():
                position = next(i for i, row in enumerate(rows) if row["pair"] == target)
                record["methods"][method] = {
                    "target_rank": position + 1,
                    "target_z": rows[position]["z"],
                    "target_rung": rows[position]["rung"],
                    "max_z": rows[0]["z"],
                    "top3": [r["pair"] for r in rows[:3]],
                    "top10": rows[:10],
                    "all_z": {r["pair"]: r["z"] for r in rows},
                }
            if do_refit:
                validation, vy, _, _ = sample(case, strength, replicate, phase, 1, 2000)
                refits = {
                    "baseline": {"validation_loss": loss(vy, model.predict(validation), family)}
                }
                fitted_models = {"baseline": model}
                union = sorted({p for rows in record["methods"].values() for p in rows["top3"]})
                tick = time.perf_counter()
                for pair in union:
                    try:
                        fitted = fit(train, y, family, [tuple(pair.split(":"))])
                        refits[pair] = {
                            "validation_loss": loss(vy, fitted.predict(validation), family)
                        }
                        fitted_models[pair] = fitted
                    except Exception as error:
                        refits[pair] = {"error": f"{type(error).__name__}: {error}"}
                record["refit_seconds"] = time.perf_counter() - tick
                record["refits"] = refits
                winners = {
                    method: select_by_validation(rows["top3"], refits)
                    for method, rows in record["methods"].items()
                }
                test, ty, tmean, _ = sample(case, strength, replicate, phase, 2, 8000)
                outcomes, measured = score_frozen_selections(
                    winners, fitted_models, test, ty, tmean, family
                )
                for pair, metrics in measured.items():
                    refits[pair].update(metrics)
                for method, rows in record["methods"].items():
                    rows.update(outcomes[method])
            record["status"] = "ok"
        except Exception:
            record["error"] = traceback.format_exc()
        record["warnings"] = sorted({str(w.message) for w in caught})
    record["elapsed_seconds"] = time.perf_counter() - started
    record["worker_peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return record


def source_hash():
    digest = hashlib.sha256()
    root = Path(superglm.__file__).parent
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "study"), default="pilot")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--strengths", nargs="+", type=float, default=list(STRENGTHS))
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--calibration", type=int, default=200)
    parser.add_argument("--audit", type=int, default=100)
    parser.add_argument("--refit-replicates", type=int, default=20)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--output", type=Path, default=Path(".benchmark-artifacts/psst-detection-study")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "pilot":
        tasks = [(case, 0.08, 0, "pilot", True) for case in args.cases]
    else:
        tasks = [
            (case, strength, replicate, "signal", replicate < args.refit_replicates)
            for case in args.cases
            for strength in args.strengths
            for replicate in range(args.replicates)
        ]
        for phase, count in (("calibration", args.calibration), ("audit", args.audit)):
            tasks.extend(
                (case, 0.0, replicate, phase, False)
                for case in NULL_CASES
                for replicate in range(count)
            )
    manifest = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "package_source_sha256": source_hash(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(
            Path(__file__).with_name("psst_detection_protocol.md").read_bytes()
        ).hexdigest(),
        "base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "threads": {
            k: os.environ.get(k)
            for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS")
        },
        "tasks": len(tasks),
    }
    manifest_path = args.output / f"{args.phase}-manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("Existing manifest differs; use a new output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    output = args.output / f"{args.phase}.jsonl"
    done = (
        {json.loads(line)["id"] for line in output.read_text().splitlines()}
        if output.exists()
        else set()
    )
    tasks = [t for t in tasks if f"{t[3]}/{t[0]}/{t[1]:g}/{t[2]}" not in done]
    started = time.perf_counter()
    print(f"{len(tasks)} remaining datasets, {args.workers} workers", flush=True)
    with output.open("a") as stream, ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_trial, task) for task in tasks]
        failures = 0
        for i, future in enumerate(as_completed(futures), 1):
            record = future.result()
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            failures += record["status"] != "ok"
            if i % 10 == 0 or i == len(tasks) or record["status"] != "ok":
                print(
                    f"{i}/{len(tasks)} complete; failures={failures}; elapsed={time.perf_counter() - started:.1f}s; last={record['id']}",
                    flush=True,
                )
                if record["status"] != "ok":
                    print(record["error"], flush=True)


if __name__ == "__main__":
    main()
