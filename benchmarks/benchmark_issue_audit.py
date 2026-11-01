"""Seeded complete-fit receipts; run baseline, head, then baseline serially.

Data/model construction and one warmup are excluded from fit timings. Diagnostics
come from a separate untimed fit. Peak RSS is cumulative for the whole process;
pickle size covers the retained model graph, not native/global caches or heap size.
Write receipts outside the repository. This script makes no speedup judgement.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import pickle
import platform
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, gammaln, xlogy


def scenarios(package, rows):
    rng = np.random.default_rng(431)
    values = rng.uniform(-1.0, 1.0, size=(2 * rows, 6))
    numeric = pd.DataFrame(values, columns=[f"x{k}" for k in range(6)])
    y = rng.binomial(1, 0.48 + values @ np.array([0.06, -0.04, 0.03, -0.02, 0.01, 0.05]))
    for link in ("logit", "identity"):
        yield f"binomial_{link}", numeric, y, None, "binomial", link
    x = values[:, 0]
    frame = pd.DataFrame({"x": x})
    offset = 3.0 + 2.0 * x + 0.04 * np.sin(3.0 * x)
    yield (
        "binomial_identity_affine_offset",
        frame,
        rng.binomial(1, 0.45 + 0.15 * x),
        offset,
        "binomial",
        "identity",
    )
    codes = np.resize(np.arange(12), 2 * rows)
    rng.shuffle(codes)
    frame = pd.DataFrame({"x": x, "z": values[:, 1], "g": codes.astype(str)})
    mean = np.exp(0.2 + 0.15 * x + 0.25 * np.sin(3 * values[:, 1]) + 0.02 * codes)
    yield "poisson_mixed", frame, rng.poisson(mean), None, "poisson", "log"
    x = np.tile(np.arange(5) / 4.0, max(1, rows // 5))
    offset = 3.0 + 2.0 * x + np.tile([0.01, 0.01, 0.01, 0.99, 0.01], len(x) // 5)
    yield (
        "binomial_identity_phase_one",
        pd.DataFrame({"x": x}),
        np.tile([0, 0, 0, 0, 1], len(x) // 5),
        offset,
        "binomial",
        "identity",
    )


def summary(values):
    values = np.asarray(values, dtype="<f8")
    finite = values[np.isfinite(values)]
    return {
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "sum": float(finite.sum()),
        "nonfinite_count": int(values.size - finite.size),
        "sha256_float64_le": hashlib.sha256(values.tobytes()).hexdigest(),
        "first_8": values[:8].tolist(),
    }


def proper_loss(y, mean, family):
    valid = np.isfinite(mean) & (mean >= 0.0)
    if family == "binomial":
        valid &= mean <= 1.0
    if not np.all(valid):
        return {"value": None, "status": "invalid_mean_domain"}
    loss = (
        -xlogy(y, mean) - xlogy(1.0 - y, 1.0 - mean)
        if family == "binomial"
        else mean - xlogy(y, mean) + gammaln(y + 1.0)
    )
    return {
        "value": float(np.mean(loss)),
        "status": "ok" if np.isfinite(loss).all() else "infinite",
    }


def peak_rss_bytes():
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value * (1 if sys.platform == "darwin" else 1024))
    except ImportError:
        return None


def run_case(package, predictor, scenario, rows, repeats):
    name, frame, y, offset, family, link = scenario
    n = min(rows, len(frame))
    X, y_train = frame.iloc[:n], y[:n].astype(float)
    train_offset = None if offset is None else offset[:n]

    def make_model():
        features = {name: package.Numeric() for name in frame}
        if family == "poisson":
            features.update(z=package.Spline(n_knots=8), g=package.Categorical(base="first"))
        return package.SuperGLM(
            family=family, link=link, selection_penalty=0, spline_penalty=0.2, features=features
        )

    def fit(model, diagnostics=False):
        return model.fit(X, y_train, offset=train_offset, record_diagnostics=diagnostics)

    timings = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit(make_model())
        for _ in range(repeats):
            model = make_model()
            start = time.perf_counter()
            fit(model)
            timings.append(time.perf_counter() - start)
        retained_bytes = len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
        eta = predictor(model._dm, model._solver_result, train_offset)
        mean = eta if link == "identity" else expit(eta) if link == "logit" else np.exp(eta)
        public_mean = model.predict(X, offset=train_offset)
        heldout = None
        if len(frame) > n:
            test_offset = None if offset is None else offset[n:]
            heldout = proper_loss(y[n:], model.predict(frame.iloc[n:], offset=test_offset), family)
        result = model.result
        diagnostic_model = fit(make_model(), diagnostics=True)
        diagnostics = diagnostic_model.iteration_diagnostics().to_dict("records")
    return name, {
        "rows": n,
        "seconds": timings,
        "median_seconds": statistics.median(timings),
        "process_cumulative_peak_rss_bytes": peak_rss_bytes(),
        "retained_model_pickle_bytes": retained_bytes,
        "raw_training_mean": summary(mean),
        "raw_training_mean_invalid_count": int(
            np.count_nonzero(
                ~np.isfinite(mean) | (mean < 0) | ((mean > 1) & (family == "binomial"))
            )
        ),
        "public_training_mean": summary(public_mean),
        "raw_training_proper_loss": proper_loss(y_train, mean, family),
        "heldout_public_proper_loss": heldout,
        "beta": result.beta.tolist(),
        "intercept": result.intercept,
        "deviance": result.deviance,
        "edf": result.effective_df,
        "backend": result.direct_backend,
        "iterations": result.n_iter,
        "converged": result.converged,
        "termination": result.termination_reason,
        "separate_untimed_fit_iteration_diagnostics": diagnostics,
        "warnings": sorted({str(w.message) for w in caught}),
    }


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    return str(value) if isinstance(value, float) and not math.isfinite(value) else value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "src")
    parser.add_argument("--rows", type=int, default=6000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rows < 5 or args.repeats < 1:
        parser.error("rows must be at least 5 and repeats must be positive")
    source = args.source.resolve(strict=True)
    sys.path.insert(0, str(source))
    package = importlib.import_module("superglm")
    module_file = Path(package.__file__).resolve()
    if not module_file.is_relative_to(source):
        raise RuntimeError(f"Imported {module_file}, outside requested source {source}")
    predictor = importlib.import_module("superglm.solvers.mode_score").linear_predictor
    git = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    payload = {
        "source": str(source),
        "module_file": str(module_file),
        "git_head_if_available": git.stdout.strip() if git.returncode == 0 else None,
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "numba", "tabmat")
        },
        "threads": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS",
                "SUPERGLM_N_JOBS",
            )
        },
        "rows_requested": args.rows,
        "repeats": args.repeats,
        "measurement_notes": __doc__,
        "nonfinite_policy": "non-finite scalars become strings; JSON NaN is forbidden",
        "cases": {},
    }
    for scenario in scenarios(package, args.rows):
        try:
            name, record = run_case(package, predictor, scenario, args.rows, args.repeats)
        except Exception as error:
            name, record = scenario[0], {"error": f"{type(error).__name__}: {error}"}
        payload["cases"][name] = record
    args.output.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
