"""Source-bound, sequential complete-fit observation/Dirichlet receipts.

No arguments prints help. See superlss_observations_dirichlet_harness.md for
the schema, adapters, numerical comparison policy and parent run commands.
Imports of numerical libraries and superglm happen only inside workers (after
pinning threads and selecting the source). No checkout is created or modified.
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import pstats
import subprocess
import sys
import time
import traceback
import uuid
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from pathlib import Path

BASELINE = "1b63f7c59e3f351fcb194f0eb25ac4e1106f2ca7"
SCHEMA = "superlss-observations-dirichlet/v1"
SEED = 62118
ROOT = Path(__file__).resolve().parents[1]
THREAD_ENV = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
FAMILIES = {
    "gaussian": "GaussianLS",
    "gamma": "GammaLS",
    "log-normal": "LogNormalLS",
    "generalized-gamma": "GeneralizedGammaLSS",
    "gpd": "GeneralizedParetoLSS",
    "lomax": "LomaxLSS",
    "log-logistic": "LogLogisticLSS",
    "two-piece-normal": "TwoPieceNormalLSS",
    "two-piece-log-normal": "TwoPieceLogNormalLSS",
    "negative-binomial": "NegativeBinomialLS",
    "tweedie": "TweedieLSS",
    "dirichlet": "DirichletLSS",
}
NEW_SCALARS = ("two-piece-normal", "two-piece-log-normal", "negative-binomial", "tweedie")


@dataclass(frozen=True)
class Case:
    law: str
    observation: str = "complete"
    fit: str = "fixed"
    weights: str = "unit"
    form: str = "location"
    components: int = 0
    offset: str = "none"

    @property
    def baseline_supported(self):
        return self.law != "dirichlet" and (
            self.law not in NEW_SCALARS or self.observation == "complete"
        )


def case_matrix():
    cases = {}
    for law in tuple(FAMILIES)[:7]:
        cases[f"{law}-bounded"] = Case(law, observation="combined", fit="smooth")
        cases[f"{law}-bounded-frequency"] = replace(cases[f"{law}-bounded"], weights="frequency")
    for law in ("gaussian", "gamma"):
        cases[f"{law}-bounded-prior"] = replace(cases[f"{law}-bounded"], weights="prior")
    cases["log-normal-mean-bounded"] = Case("log-normal", "combined", "smooth", form="mean")
    for law in NEW_SCALARS:
        forms = ("location", "mean") if law == "two-piece-log-normal" else ("location",)
        for form in forms:
            prefix = law + ("-mean" if form == "mean" else "")
            for observation in ("complete", "exact", "mixed", "combined"):
                case = Case(law, observation, form=form)
                cases[f"{prefix}-{observation}"] = case
                cases[f"{prefix}-{observation}-frequency"] = replace(case, weights="frequency")
                if law in ("negative-binomial", "tweedie"):
                    cases[f"{prefix}-{observation}-prior"] = replace(case, weights="prior")
            cases[f"{prefix}-combined-smooth"] = Case(law, "combined", "smooth", form=form)
    for k in (2, 3, 5):
        for fit in ("intercept", "fixed", "smooth"):
            cases[f"dirichlet-k{k}-{fit}"] = Case("dirichlet", fit=fit, components=k)
        cases[f"dirichlet-k{k}-fixed-frequency"] = Case(
            "dirichlet", weights="frequency", components=k
        )
    for offset in ("columns", "arrays"):
        cases[f"gaussian-offset-{offset}"] = Case("gaussian", offset=offset)
        cases[f"gaussian-offset-{offset}-smooth"] = Case("gaussian", fit="smooth", offset=offset)
        cases[f"dirichlet-k3-offset-{offset}"] = Case("dirichlet", components=3, offset=offset)
    return cases


CASES = case_matrix()


class MissingStageError(RuntimeError):
    """A required public interface is not integrated in the selected source."""


def jsonable(value):
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def canonical(value):
    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_identity(source, *, expected_sha=None, require_clean=False):
    source = Path(source).resolve()
    if source.name != "src" or not (source / "superglm/__init__.py").is_file():
        raise ValueError(f"source must be a checkout's src directory: {source}")

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(source), *args], text=True, timeout=15
        ).strip()

    root = Path(git("rev-parse", "--show-toplevel")).resolve()
    if source != root / "src":
        raise ValueError("source is not the checkout's root src directory")
    sha = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--untracked-files=normal")
    if expected_sha is not None and sha != expected_sha:
        raise ValueError(f"expected source head {expected_sha}; got {sha}")
    if require_clean and dirty:
        raise ValueError(f"baseline source must be clean: {dirty}")
    hashes = {
        str(p.relative_to(root)): file_hash(p) for p in sorted((source / "superglm").rglob("*.py"))
    }
    for name in ("pyproject.toml", "uv.lock"):
        hashes[name] = file_hash(root / name)
    return {
        "path": str(source),
        "sha": sha,
        "digest": digest(hashes),
        "files": hashes,
        "dirty": dirty,
    }


def save_arrays(path, arrays):
    import numpy as np

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.asarray(value) for key, value in arrays.items()}
    np.savez_compressed(path, **arrays)
    return {
        "path": str(path),
        "sha256": file_hash(path),
        "arrays": {
            key: {
                "shape": list(a.shape),
                "dtype": str(a.dtype),
                "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            }
            for key, a in arrays.items()
        },
    }


def load_arrays(artifact):
    import numpy as np

    path = Path(artifact["path"])
    if file_hash(path) != artifact["sha256"]:
        raise ValueError(f"artifact hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def run_process(command, output, *, timeout, env, context):
    """Keep partial checkpoints and a failure row even if a worker dies before JSON."""
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError(f"refusing a stale worker receipt: {output}")
    log = output.with_suffix(".log")
    result = dict(context, command=list(command), log=str(log), observations=[])
    started = time.perf_counter()
    timeout_error = None
    returncode = None
    with log.open("w") as stream:
        try:
            process = subprocess.run(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=timeout,
                check=False,
            )
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            timeout_error = f"worker exceeded {timeout:g} seconds; partial observations retained"
        except OSError as exc:
            result["error"] = str(exc)
    try:
        worker = json.loads(output.read_text())
        if not isinstance(worker, dict) or not isinstance(worker.get("observations"), list):
            raise ValueError("invalid worker receipt")
        result.update(worker)
    except (OSError, ValueError) as exc:
        result.update(status="failed", error=f"missing or malformed worker receipt: {exc}")
    result.update(returncode=returncode, process_seconds=time.perf_counter() - started)
    if timeout_error:
        result.update(status="timed_out", error=timeout_error)
    elif returncode != 0:
        if result.get("status") not in ("refused", "missing_stage"):
            result["status"] = "failed"
        result.setdefault("error", f"worker exit code {returncode}")
    elif result.get("status") == "running":
        result.update(status="failed", error="worker exited without a final receipt")
    write_json(output, result)
    return result


def compare_records(candidate, control, *, kind="warm", control_kind=None, execution_check=True):
    """A mismatch retains deltas but cannot silently become an accepted time ratio."""
    import numpy as np

    mismatches = [
        key
        for key in ("data", "weights", "spec", "environment", "machine", "threads", "harness")
        if not candidate.get("identity", {}).get(key)
        or candidate["identity"][key] != control.get("identity", {}).get(key)
    ]
    if not candidate.get("session") or candidate["session"] != control.get("session"):
        mismatches.append("session")
    for record in (candidate, control):
        if record.get("role") == "baseline" and record.get("source", {}).get("sha") != BASELINE:
            mismatches.append("baseline_source")
    if execution_check and candidate.get("execution") != control.get("execution"):
        mismatches.append("execution")
    out = dict(
        comparable=False,
        identity_mismatches=sorted(set(mismatches)),
        deltas={},
        time_ratio=None,
        numerical_match=False,
        backend_match=False,
        convergence_match=False,
    )
    a = next((o for o in candidate.get("observations", []) if o["kind"] == kind), None)
    b = next(
        (o for o in control.get("observations", []) if o["kind"] == (control_kind or kind)), None
    )
    if a is None or b is None:
        return dict(out, error="missing fit observation; failed/timed-out workers remain in runs")
    out["backend_match"] = bool(a.get("backend")) and a.get("backend") == b.get("backend")
    out["convergence_match"] = all(
        o.get("converged") is True and o.get("smoothing_converged") is not False for o in (a, b)
    )
    out["public_match"] = all(o.get("public_status") == "completed" for o in (a, b))
    try:
        left, right = load_arrays(a["numerics"]), load_arrays(b["numerics"])
    except (OSError, ValueError, KeyError) as exc:
        return dict(out, error=f"numerical artifact unavailable: {exc}")
    numerical_match = bool(left) and left.keys() == right.keys()
    condition = max(float(a.get("condition", math.inf)), float(b.get("condition", math.inf)))
    out["conditioning_supported"] = math.isfinite(condition) and condition < 1e10
    for key in sorted(left.keys() | right.keys()):
        if key not in left or key not in right or left[key].shape != right[key].shape:
            out["deltas"][key] = {"match": False, "reason": "missing or different shape"}
            numerical_match = False
            continue
        x, y = left[key], right[key]
        scale = float(np.max(np.abs(y), initial=0))
        # Solver stopping dominates binary64 on these well-conditioned fixtures.
        # Ill-conditioned endpoints are disclosed, never licensed by a huge tolerance.
        relative = 2e-5 if "covariance" in key or "se" in key else 2e-6
        relative = max(relative, min(1e-3, 64 * np.finfo(float).eps / 2 * condition))
        tolerance = relative * max(1.0, scale)
        finite = bool(np.isfinite(x).all() and np.isfinite(y).all())
        delta = float(np.max(np.abs(x - y), initial=0)) if finite else math.inf
        matched = finite and delta <= tolerance
        out["deltas"][key] = dict(
            max_absolute=delta, scale=scale, tolerance=tolerance, match=matched
        )
        numerical_match &= matched
    for key, relative in (("objective", 5e-8), ("edf", 5e-6), ("rank", 0.0)):
        x, y = float(a.get(key, math.nan)), float(b.get(key, math.nan))
        delta, tolerance = abs(x - y), relative * max(1.0, abs(y))
        matched = math.isfinite(delta) and delta <= tolerance
        out["deltas"][key] = dict(max_absolute=delta, tolerance=tolerance, match=matched)
        numerical_match &= matched
    out["numerical_match"] = bool(numerical_match)
    out["comparable"] = bool(
        not mismatches
        and numerical_match
        and (out["backend_match"] or not execution_check)
        and out["convergence_match"]
        and out["public_match"]
        and out["conditioning_supported"]
        and all(r.get("status") == "completed" for r in (candidate, control))
    )
    if (
        out["comparable"]
        and execution_check
        and a.get("seconds") is not None
        and b.get("seconds", 0) > 0
    ):
        out["time_ratio"] = a["seconds"] / b["seconds"]
    return out


def retained_memory(root):
    """Union owners, then classify aliases; a later cache alias must count too."""
    import numpy as np
    import pandas as pd

    seen, owners = set(), {}

    def visit(obj, path, cached=False):
        cached |= any(
            word in path.lower()
            for word in ("cache", "prepared", "log_shares", "bounds", "likelihood_plan", "table")
        )
        if isinstance(obj, np.ndarray):
            owner = obj
            while isinstance(getattr(owner, "base", None), np.ndarray):
                owner = owner.base
            buffer = getattr(owner, "base", None)
            if isinstance(buffer, memoryview):
                buffer = buffer.obj
            if isinstance(buffer, (bytes, bytearray)):
                owner, size = buffer, len(buffer)
            else:
                size = owner.nbytes
            record = owners.setdefault(id(owner), {"bytes": size, "cache": False, "paths": []})
            record["cache"] |= cached
            if len(record["paths"]) < 4:
                record["paths"].append(path)
            return
        key = (id(obj), cached)
        if key in seen:
            return
        seen.add(key)
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            # Inspect existing blocks, never to_numpy(), which can allocate a
            # consolidated copy and count memory the fitted model does not own.
            for j, block in enumerate(obj._mgr.blocks):
                if isinstance(block.values, np.ndarray):
                    visit(block.values, path + f".block{j}", cached)
        elif isinstance(obj, Mapping):
            for name, value in obj.items():
                visit(value, path + "." + str(name), cached)
        elif isinstance(obj, (tuple, list)):
            for i, value in enumerate(obj):
                visit(value, path + f".{i}", cached)
        elif is_dataclass(obj) and not isinstance(obj, type):
            for field in fields(obj):
                visit(getattr(obj, field.name), path + "." + field.name, cached)
        elif (
            not isinstance(obj, type)
            and type(obj).__module__.startswith(("superglm", "scipy"))
            and hasattr(obj, "__dict__")
        ):
            visit(vars(obj), path, cached)

    visit(root, "model")
    buffers = sorted(owners.values(), key=lambda r: r["bytes"], reverse=True)
    return dict(
        unique_buffer_bytes=sum(r["bytes"] for r in buffers),
        cache_buffer_bytes=sum(r["bytes"] for r in buffers if r["cache"]),
        buffer_count=len(buffers),
        largest_buffers=buffers[:20],
        scope="unique reachable NumPy backing stores, including bytes and views; cache is a path-tagged subset",
        excluded="Python object headers, non-NumPy native allocations and process-global JIT caches; see process RSS",
    )


def data_identity(data):
    import numpy as np

    def array_hash(value):
        a = np.asarray(value)
        return {
            "shape": a.shape,
            "dtype": str(a.dtype),
            "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
        }

    keys = ("x", "y", "latent", "lower", "upper", "side", "offset_a", "offset_b", "truth")
    return {
        "data": digest(
            {**{k: array_hash(data[k]) for k in keys}, "columns": data["columns"], "seed": SEED}
        ),
        "weights": digest({"semantics": data["semantics"], "values": array_hash(data["weights"])}),
    }


def make_data(case, rows, *, grid="repeat"):
    """Independent synthetic draws; exact-repeat covariates and continuous grid differ explicitly."""
    import numpy as np
    from scipy.special import ndtr, ndtri

    rng = np.random.default_rng(SEED)
    weights = rng.integers(1, 4, rows).astype(float) if case.weights != "unit" else np.ones(rows)
    prior = weights if case.weights == "prior" else np.ones(rows)
    x = np.resize(np.linspace(-1, 1, 16), rows) if grid == "repeat" else rng.uniform(-1, 1, rows)
    varying = np.zeros(rows) if case.fit == "intercept" else 0.25 * np.sin(2 * x)
    oa, ob = 0.08 * (1 + x), -0.04 * x
    offset = oa + ob if case.offset != "none" else np.zeros(rows)
    law = case.law
    real = law in ("gaussian", "two-piece-normal")
    mean = np.exp(0.4 + varying + offset)
    location = 0.3 + varying + offset
    if law == "dirichlet":
        alpha = np.linspace(1.4, 3.2, case.components)[None, :] * np.exp(
            varying[:, None] * np.linspace(-1, 1, case.components)
        )
        alpha[:, 0] *= np.exp(offset)
        latent = np.array([rng.dirichlet(a) for a in alpha])
        truth = alpha
    else:
        if real:
            values = (location, np.full(rows, 0.65))
        elif law == "gamma":
            values = (mean, np.full(rows, 0.45))
        elif law == "negative-binomial":
            values = (mean, np.full(rows, 2.5))
        elif law == "tweedie":
            values = (mean, np.full(rows, 1.4), np.full(rows, 1.5))
        elif law in ("lomax", "log-logistic"):
            values = (mean, np.full(rows, 3.5))
        elif law == "gpd":
            values = (mean / 3.5, np.full(rows, 1 / 3.5))
        elif law == "generalized-gamma":
            values = (0.4 + varying, np.full(rows, 0.55), np.full(rows, 0.6))
        elif law == "log-normal":
            values = (
                mean if case.form == "mean" else np.log(mean) - 0.65**2 / 2,
                np.full(rows, 0.65),
            )
        else:
            a, b = 0.65 * (1 - 0.25), 0.65 * (1 + 0.25)
            loading = (1 - 0.25) * np.exp(a * a / 2) * ndtr(-a) + (1 + 0.25) * np.exp(
                b * b / 2
            ) * ndtr(b)
            values = (
                np.exp(location) * loading if case.form == "mean" else location,
                np.full(rows, 0.65),
            )
        if law.startswith("two-piece"):
            values = (*values, np.full(rows, 0.25))
        truth = np.column_stack(values)

        def draw(indices):
            n = len(indices)
            w = prior[indices]
            m, mu = mean[indices], location[indices]
            if law == "gaussian":
                return rng.normal(mu, 0.65 / np.sqrt(w))
            if law == "gamma":
                return rng.gamma(w / 0.45**2, m * 0.45**2 / w)
            if law in ("log-normal", "generalized-gamma"):
                if law == "log-normal":
                    return rng.lognormal(np.log(m) - 0.65**2 / 2, 0.65)
                q, scale = 0.6, 0.55
                return np.exp(0.4 + varying[indices]) * (rng.gamma(1 / q**2, q**2, n)) ** (
                    scale / q
                )
            if law in ("lomax", "gpd"):
                return m * rng.pareto(3.5, n)
            if law == "log-logistic":
                u = rng.uniform(0.001, 0.999, n)
                return m * (u / (1 - u)) ** (1 / 3.5)
            if law.startswith("two-piece"):
                skew, scale = 0.25, 0.65
                u = rng.uniform(0.0001, 0.9999, n)
                left = u < (1 - skew) / 2
                z = np.where(
                    left,
                    (1 - skew) * ndtri(u / (1 - skew)),
                    -(1 + skew) * ndtri((1 - u) / (1 + skew)),
                )
                value = mu + scale * z
                return value if real else np.exp(value)
            if law == "negative-binomial":
                return rng.negative_binomial(2.5 * w, 2.5 / (2.5 + m)) / w
            if law == "tweedie":
                phi, power = 1.4, 1.5
                counts = rng.poisson(w * m ** (2 - power) / (phi * (2 - power)))
                return rng.gamma(
                    counts * (2 - power) / (power - 1), phi * (power - 1) * m ** (power - 1) / w
                )
            raise ValueError(law)

        latent = draw(np.arange(rows))
    lower, upper = np.full(rows, -np.inf), np.full(rows, np.inf)
    if case.observation in ("exact", "combined"):
        # Independent lower-only, upper-only, and two-sided windows. Zero atoms
        # remain in rows whose lower endpoint is -inf; (0, upper] excludes them.
        lower[np.arange(rows) % 4 != 0] = (
            -1.6 if real else (0.0 if law in ("negative-binomial", "tweedie") else 0.15)
        )
        upper[np.arange(rows) % 4 != 1] = 2.6 if real else 8.0
        for _ in range(10000):
            bad = np.flatnonzero((latent <= lower) | (latent > upper))
            if not len(bad):
                break
            latent[bad] = draw(bad)
        else:
            raise RuntimeError("synthetic truncation rejection budget exhausted")
    y = latent.copy()
    side = np.zeros(rows, dtype=int)
    if case.observation in ("mixed", "combined"):
        low, high = (-0.5, 1.3) if real else (0.5, 3.0)
        side[(latent > lower) & (latent < low)] = -1
        side[latent > high] = 1
        # Retain some exact zeros/counts alongside a left-censored atom.
        side[np.arange(rows) % 3 == 0] = 0
        y = np.where(side < 0, low, np.where(side > 0, high, latent))
    data = dict(
        x=x,
        y=y,
        latent=latent,
        lower=lower,
        upper=upper,
        side=side,
        weights=weights,
        offset_a=oa,
        offset_b=ob,
        truth=truth,
        columns=[f"share_{j}" for j in range(case.components)],
        semantics="frequency" if case.weights == "frequency" else "prior",
    )
    data["identity"] = data_identity(data)
    return data


def family_for(case):
    import superglm

    cls = getattr(superglm, FAMILIES[case.law], None)
    if cls is None:
        raise MissingStageError(f"public {FAMILIES[case.law]} export is unavailable")
    if case.law == "dirichlet":
        return cls(components=tuple(f"share_{j}" for j in range(case.components)))
    if case.law in ("log-normal", "generalized-gamma", "two-piece-log-normal"):
        return cls(parametrisation=case.form)
    return cls()


def frame_for(data):
    import pandas as pd

    return pd.DataFrame(
        {"x": data["x"], "offset_a": data["offset_a"], "offset_b": data["offset_b"]},
        index=pd.RangeIndex(70, 70 + len(data["x"]), name="synthetic_row"),
    )


def response_for(data, frame):
    import pandas as pd

    return (
        pd.DataFrame(data["y"], columns=data["columns"], index=frame.index)
        if data["columns"]
        else data["y"]
    )


def bound_options(case, data):
    import numpy as np

    options = {}
    if case.observation in ("exact", "combined"):
        options["truncation"] = (data["lower"], data["upper"])
    if case.observation in ("mixed", "combined"):
        options["censored"] = np.where(
            data["side"] < 0, "left", np.where(data["side"] > 0, "right", "none")
        )
    return options


def offset_adapter(case, role):
    """Only the approved baseline can use the removed fitting keyword."""
    from superglm import SuperLSS, bind_predictor

    if case.offset == "none":
        return "none"
    if role == "baseline":
        return "baseline-1b63-fit-mapping: sum(offset_a, offset_b), coefficient=1"
    if "offset" not in inspect.signature(bind_predictor).parameters:
        raise MissingStageError("Task9: bind_predictor(..., offset=...) is not integrated")
    for method in (SuperLSS.fit, SuperLSS.fit_reml):
        if "offsets" in inspect.signature(method).parameters:
            raise MissingStageError(
                "Task9: public fit/fit_reml still expose the removed offsets keyword"
            )
    return f"candidate-predictor-{case.offset}: sum(offset_a, offset_b), coefficient=1"


def complete_fit(case, data, config, role):
    from superglm import SuperLSS, bind_predictor, s

    offset_adapter(case, role)  # a candidate with missing Task9 never falls back
    family = family_for(case)
    predictors, lambdas = [], {}
    for j, parameter in enumerate(family.parameters):
        varying = case.fit != "intercept" and (j == 0 or case.law == "dirichlet")
        terms = (s("x", kind="cr", k=5 + j % 2),) if varying else ()
        declaration = {}
        if j == 0 and case.offset != "none" and role != "baseline":
            declaration["offset"] = (
                ["offset_a", "offset_b"]
                if case.offset == "columns"
                else [data["offset_a"], data["offset_b"]]
            )
        predictors.append(bind_predictor(family, parameter.name, *terms, **declaration))
        if varying:
            lambdas[f"{parameter.name}:x#wiggle"] = 0.8
    model = SuperLSS(
        family,
        *predictors,
        weight_semantics=data["semantics"],
        discrete=config["discrete"],
        n_bins=config["n_bins"],
    )
    frame = frame_for(data)
    response = response_for(data, frame)
    options = dict(
        sample_weight=data["weights"],
        retain_rows=not config["compact"],
        max_inner_iter=config["max_inner_iter"],
        inner_tol=1e-7,
        **bound_options(case, data),
    )
    if case.offset != "none" and role == "baseline":
        options["offsets"] = {family.parameters[0].name: data["offset_a"] + data["offset_b"]}
    if case.fit == "smooth":
        model.fit_reml(
            frame,
            response,
            initial_lambda=config["initial_lambda"],
            max_reml_iter=config["max_reml_iter"],
            **options,
        )
    else:
        model.fit(frame, response, lambdas=lambdas, **options)
    return model, frame


def prediction_options(case, data, family, role):
    if case.offset == "none" or (case.offset == "columns" and role != "baseline"):
        return {}
    # Prediction values replace the entire offset sum for new rows. No training
    # array reuse, implicit log, or fitted coefficient participates.
    return {"offsets": {family.parameters[0].name: data["offset_a"] + data["offset_b"]}}


def public_outputs(model, frame, case, data, role):
    import numpy as np
    import pandas as pd

    output, checks = {}, {}
    options = prediction_options(case, data, model.family, role)
    parameters = model.predict_parameters(frame, **options)
    output["parameters"] = parameters.to_numpy()
    link, link_se = model.predict_link(frame, se_fit=True, **options)
    output.update(link=link.to_numpy(), link_se=link_se.to_numpy())
    prediction = model.predict(frame, **options)
    if case.law == "dirichlet":
        if not isinstance(prediction, pd.DataFrame):
            raise MissingStageError(
                "Task6: public Dirichlet predict must return a labeled DataFrame; no internal-mean substitution"
            )
        if list(prediction.columns) != data["columns"] or not prediction.index.equals(frame.index):
            raise MissingStageError(
                "Task6: Dirichlet prediction columns/index do not match the declared components/input"
            )
        output["prediction"] = prediction.to_numpy()
        checks["simplex_error"] = float(np.max(np.abs(output["prediction"].sum(axis=1) - 1)))
        if np.any(output["prediction"] <= 0) or checks["simplex_error"] > 1e-12:
            raise ValueError("public Dirichlet predictions violate the positive simplex")
        # Response covariance is distinct from coefficient/posterior covariance.
        output["response_covariance"] = model.family.response_covariance(output["parameters"])
    else:
        output["prediction"] = np.asarray(prediction)
    scores = model.scores(
        frame,
        response_for(data, frame),
        which=("log",),
        sample_weight=data["weights"],
        **options,
        **bound_options(case, data),
    )
    if scores.shape != (len(frame), 1) or not scores.index.equals(frame.index):
        raise ValueError(
            "public joint log score must have one row per input row with preserved index"
        )
    output["log_score"] = scores.to_numpy().ravel()
    checks["log_score_sum"] = float(output["log_score"].sum())
    return output, checks


def collect_fit(model, frame, case, data, role, artifact, *, memory=False):
    import numpy as np

    fitted = model._require_fitted()
    state = fitted.fit_state
    result = state.solver_result
    published = model.result_
    covariance = model.covariance_
    arrays = {
        "coefficients": published.coefficients,
        "covariance": covariance,
        "coefficient_se": np.sqrt(np.maximum(np.diag(covariance), 0.0)),
    }
    # Public prediction/scoring failures do not erase the completed fit or its
    # convergence evidence. They explicitly prevent lifecycle acceptance.
    record = dict(
        status="completed",
        converged=bool(result.converged),
        stop_reason=str(result.convergence_reason),
        smoothing_converged=None,
        public_status="completed",
        backend=result.execution_backend_identifier,
        rank=int(published.rank),
        edf=float(published.total_effective_df),
        objective=float(published.penalized_log_likelihood),
        log_likelihood=float(published.log_likelihood),
        condition=float(np.linalg.cond(covariance)),
        resolved_chunk_size=result.resolved_chunk_size,
        work={
            "coefficient_iterations": int(result.iterations),
            "backtracking_steps": int(result.backtracking_steps),
        },
        coefficient_names=published.coefficient_names,
        parameter_names=published.parameter_names,
        warnings=list(published.warnings),
    )
    smoothing = state.smoothing
    if smoothing is not None:
        record["smoothing_converged"] = bool(smoothing.converged)
        record["smoothing"] = {
            key: jsonable(getattr(smoothing, key, None))
            for key in (
                "convergence_reason",
                "matched_certified",
                "objective",
                "iterations",
                "lambdas",
                "newton_iterations",
                "bfgs_fallback_iterations",
                "terminal_projected_gradient_norm",
                "terminal_gradient_certificate",
                "smoothing_hessian_certificate",
            )
        }
        record["work"].update(
            coefficient_fits=len(smoothing.coefficient_fits),
            total_coefficient_iterations=sum(f.iterations for f in smoothing.coefficient_fits),
        )
        arrays["smoothing_parameters"] = np.array(
            [v for _, v in sorted(published.smoothing_parameters.items())]
        )
    null = state.null_model
    record["null_fit"] = {
        key: jsonable(getattr(null, key, None))
        for key in ("converged", "convergence_reason", "objective", "iterations", "rank")
    }
    try:
        record["diagnostic"] = model.diagnose().to_dict()
    except Exception as exc:
        record["diagnostic"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    snapshot = getattr(model, "_fit_phase_snapshot", None)
    record["phases"] = snapshot.as_dict() if snapshot is not None else None
    if memory:
        record["memory"] = {
            **retained_memory(model),
            **rss(),
            "measured_before_public_prediction": True,
        }
    profiler = cProfile.Profile()
    try:
        with profiler:
            public, checks = public_outputs(model, frame, case, data, role)
        arrays.update(public)
        record["public_checks"] = checks
        record["public_checks"]["log_score_likelihood_delta"] = (
            checks["log_score_sum"] + record["log_likelihood"]
        )
        if abs(record["public_checks"]["log_score_likelihood_delta"]) > 2e-8 * (
            1 + abs(record["log_likelihood"])
        ):
            raise ValueError("public log scores do not sum to the fitted normalized likelihood")
        if any(not np.isfinite(value).all() for value in public.values()):
            raise ValueError(
                "public output contains a nonfinite value; inspect moment/range disclosures"
            )
    except Exception as exc:
        record.update(
            public_status="missing_stage" if isinstance(exc, MissingStageError) else "failed",
            public_error=f"{type(exc).__name__}: {exc}",
        )
    forbidden = ("fit_dense_distributional", "fit_dense_fixed_lambda", "fit_distributional_efs")
    record["prediction_optimizer_calls"] = sum(
        values[1]
        for (_, _, name), values in pstats.Stats(profiler).stats.items()
        if name in forbidden
    )
    if record["prediction_optimizer_calls"]:
        record.update(
            public_status="failed",
            public_error="public prediction/scoring ran a training optimizer",
        )
    record["numerics"] = save_arrays(artifact, arrays)
    return record


def rss():
    import psutil

    info = psutil.Process().memory_info()
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = int(peak if sys.platform == "darwin" else peak * 1024)
        method = "getrusage process high-water mark"
    except ImportError:
        peak = getattr(info, "peak_wset", None)
        method = (
            "Windows peak_wset" if peak is not None else "unavailable (current RSS is not peak)"
        )
    return {
        "rss_bytes": info.rss,
        "peak_rss_bytes": peak,
        "peak_method": method,
        "peak_scope": "whole fresh process including import/JIT; not an incremental allocation estimate",
    }


def fingerprints():
    import numba
    import psutil
    from threadpoolctl import threadpool_info

    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "numpy",
            "scipy",
            "numba",
            "pandas",
            "tabmat",
            "narwhals",
            "scikit-learn",
            "joblib",
            "threadpoolctl",
            "psutil",
        )
    }
    pools = [
        {
            key: pool.get(key)
            for key in (
                "user_api",
                "internal_api",
                "version",
                "architecture",
                "threading_layer",
                "num_threads",
            )
        }
        for pool in threadpool_info()
    ]
    environment = dict(
        python=sys.version,
        executable=sys.executable,
        versions=versions,
        pools=sorted(pools, key=canonical),
    )
    process = psutil.Process()
    machine = dict(
        system=platform.system(),
        release=platform.release(),
        machine=platform.machine(),
        processor=platform.processor(),
        node=platform.node(),
        logical_cpus=os.cpu_count(),
        physical_cpus=psutil.cpu_count(logical=False),
        ram_bytes=psutil.virtual_memory().total,
        affinity=process.cpu_affinity() if hasattr(process, "cpu_affinity") else None,
    )
    if Path("/proc/cpuinfo").is_file():
        machine["cpu_model"] = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            None,
        )
    threads = dict(
        env={key: os.environ.get(key) for key in THREAD_ENV},
        numba=numba.get_num_threads(),
        pools=pools,
    )
    if threads["numba"] != 1 or any(p["num_threads"] != 1 for p in pools):
        raise ValueError("requested single-thread execution was not honored")
    return environment, machine, threads


def mathematical_spec(case, config):
    return dict(
        schema=SCHEMA,
        case=asdict(case),
        seed=SEED,
        basis="cr, intercept, k=5+j%2; first scalar/all Dirichlet",
        fixed_lambda=0.8,
        max_inner_iter=config["max_inner_iter"],
        inner_tol=1e-7,
        max_reml_iter=config["max_reml_iter"],
        reml_tol=1e-6,
        outer="family default: efs for two-piece, efs+finish otherwise",
        offsets="link-scale sum, coefficient one",
    )


def profile_summary(profiler):
    stats = pstats.Stats(profiler)
    entries = []
    for (filename, line, name), (primitive, calls, own, cumulative, _) in stats.stats.items():
        filename = filename.replace("\\", "/")
        if "/src/superglm/" not in filename:
            continue
        short = "superglm/" + filename.rsplit("/src/superglm/", 1)[1]
        entries.append(
            dict(
                function=f"{short}:{line}:{name}",
                calls=calls,
                primitive_calls=primitive,
                self_seconds=own,
                cumulative_seconds=cumulative,
            )
        )
    return dict(
        top=sorted(entries, key=lambda r: r["cumulative_seconds"], reverse=True)[:30],
        dispatch=[
            e
            for e in entries
            if any(word in e["function"] for word in ("/kernels/", "/assembly", "/chunks.py"))
        ],
        scope="Python and exposed compiled entry points; native inner loops are not counted",
    )


def reference_theta(case, eta):
    """Independent links for the explicitly moderate small-reference fixtures."""
    import numpy as np
    from scipy.special import expit

    if case.law in ("dirichlet", "negative-binomial"):
        return np.exp(eta)
    if case.law == "tweedie":
        return np.array([np.exp(eta[0]), np.exp(eta[1]), 1.05 + 0.9 * expit(eta[2])])
    first = np.exp(eta[0]) if case.form == "mean" else eta[0]
    return np.array([first, 0.01 + np.exp(eta[1]), 0.9 * np.tanh(eta[2] / 2)])


def reference_rows(case, data, theta):
    """Independent SciPy densities/windows, never a production CDF as oracle.

    Reference envelope only: moderate tails, NB finite endpoints below 128/w,
    Tweedie Poisson rate <=30 and 240 terms (omitted mass <1e-120). Not a second
    general-purpose likelihood engine or numerical admission certificate.
    """
    import numpy as np
    from scipy import special, stats

    y, lo, hi, side = (data[key] for key in ("y", "lower", "upper", "side"))
    w = data["weights"] if data["semantics"] == "prior" else np.ones(len(y))
    mass = data["weights"] if data["semantics"] == "frequency" else np.ones(len(y))
    if case.law == "dirichlet":
        return mass * (
            special.gammaln(theta.sum()) - special.gammaln(theta).sum() + np.log(y) @ (theta - 1)
        )
    if case.law == "negative-binomial":
        mu, size = theta
        size = size * w
        probability = theta[1] / (theta[1] + mu)

        def count_at(t):
            counts = np.arange(129.0)[:, None]
            if np.any(np.isfinite(t) & (t >= 128 / w)):
                raise ValueError(
                    "independent NB reference finite endpoint exceeds its stated envelope"
                )
            return np.where(np.isposinf(t), np.inf, ((counts / w) <= t).sum(axis=0) - 1)

        def window(a, b):
            aa, bb = count_at(a), count_at(b)
            cdf = stats.nbinom.cdf(bb, size, probability) - stats.nbinom.cdf(aa, size, probability)
            sf = stats.nbinom.sf(aa, size, probability) - stats.nbinom.sf(bb, size, probability)
            return np.where(aa > mu * w, sf, cdf)

        density = stats.nbinom.logpmf(np.rint(y * w), size, probability)
    elif case.law == "tweedie":
        mu, phi, power = theta
        rate = w * mu ** (2 - power) / (phi * (2 - power))
        if np.max(rate) > 30:
            raise ValueError("independent Tweedie reference rate exceeds 30")
        scale = phi * (power - 1) * mu ** (power - 1) / w
        j = np.arange(1.0, 241.0)[:, None]
        shape = j * (2 - power) / (power - 1)
        log_mass = j * np.log(rate) - rate - special.gammaln(j + 1)

        def window(a, b):
            aa, bb = np.maximum(a, 0) / scale, np.maximum(b, 0) / scale
            cdf = special.gammainc(shape, bb) - special.gammainc(shape, aa)
            sf = special.gammaincc(shape, aa) - special.gammaincc(shape, bb)
            return (np.exp(log_mass) * np.where(aa > shape, sf, cdf)).sum(axis=0) + np.where(
                (a < 0) & (b >= 0), np.exp(-rate), 0
            )

        positive = np.maximum(y, 1e-100)
        density = special.logsumexp(
            log_mass
            + (shape - 1) * np.log(positive)
            - positive / scale
            - special.gammaln(shape)
            - shape * np.log(scale),
            axis=0,
        )
        density = np.where(y == 0, -rate, density)
    else:
        first, scale, skew = theta
        location = first
        log_response = case.law == "two-piece-log-normal"
        if case.form == "mean":
            a, b = scale * (1 - skew), scale * (1 + skew)
            loading = (1 - skew) * np.exp(a * a / 2) * special.ndtr(-a) + (1 + skew) * np.exp(
                b * b / 2
            ) * special.ndtr(b)
            location = np.log(first) - np.log(loading)

        def transform(t):
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(t > 0, np.log(t), -np.inf) if log_response else t

        def tails(t):
            t = transform(t)
            z = (t - location) / (scale * np.where(t < location, 1 - skew, 1 + skew))
            cdf = np.where(
                t < location, (1 - skew) * special.ndtr(z), 1 - (1 + skew) * special.ndtr(-z)
            )
            sf = np.where(
                t < location, 1 - (1 - skew) * special.ndtr(z), (1 + skew) * special.ndtr(-z)
            )
            return cdf, sf

        def window(a, b):
            ca, sa = tails(a)
            cb, sb = tails(b)
            return np.where(transform(a) > location, sa - sb, cb - ca)

        t = transform(y)
        z = (t - location) / (scale * np.where(t < location, 1 - skew, 1 + skew))
        density = -np.log(scale) - np.log(2 * np.pi) / 2 - z * z / 2
        if log_response:
            density -= np.log(y)
    numerator = density.copy()
    censored = side != 0
    if censored.any():
        observed_probability = window(np.where(side > 0, y, lo), np.where(side < 0, y, hi))
        numerator[censored] = np.log(observed_probability[censored])
    return mass * (numerator - np.log(window(lo, hi)))


def small_reference(case, config, output):
    import numpy as np
    from scipy.optimize import minimize

    if case.law not in NEW_SCALARS and case.law != "dirichlet":
        return {"status": "not_applicable", "reason": "existing-law same-source baseline control"}
    small = replace(case, fit="intercept", offset="none")
    data = make_data(small, 64)
    model, frame = complete_fit(
        small, data, dict(config, discrete=False, compact=False), "candidate"
    )
    fitted = collect_fit(model, frame, small, data, "candidate", output.with_suffix(".npz"))
    coefficient = np.asarray(model.result_.coefficients)
    if case.law == "dirichlet":
        start = np.log(np.linspace(1.4, 3.2, case.components))
    elif case.law == "negative-binomial":
        start = np.log([1.5, 2.5])
    elif case.law == "tweedie":
        start = np.array([np.log(1.5), np.log(1.4), 0.0])
    else:
        start = np.array(
            [0.65 if case.form == "mean" else 0.3, np.log(0.64), 2 * np.arctanh(0.25 / 0.9)]
        )

    def objective(eta):
        try:
            with np.errstate(all="ignore"):
                if np.max(np.abs(eta)) > 15:
                    return math.inf
                value = -float(reference_rows(small, data, reference_theta(small, eta)).sum())
                return value if math.isfinite(value) else math.inf
        except ValueError:
            return math.inf

    def gradient(eta):
        h = 2e-5
        return np.array(
            [(objective(eta + h * e) - objective(eta - h * e)) / (2 * h) for e in np.eye(len(eta))]
        )

    endpoints = []
    for label, initial in (("ordinary", start), ("lower", start - 0.15), ("upper", start + 0.15)):
        result = minimize(
            objective, initial, jac=gradient, method="BFGS", options={"gtol": 1e-6, "maxiter": 180}
        )
        endpoints.append(
            dict(
                name=label,
                success=bool(result.success),
                message=str(result.message),
                iterations=result.nit,
                objective=result.fun,
                parameters=reference_theta(small, result.x),
                gradient_inf=float(np.max(np.abs(gradient(result.x)))),
                coefficients=result.x,
            )
        )
    best = min(endpoints, key=lambda r: r["objective"])
    h = 2e-4
    beta = np.asarray(best["coefficients"])
    hessian = np.column_stack(
        [(gradient(beta + h * e) - gradient(beta - h * e)) / (2 * h) for e in np.eye(len(beta))]
    )
    hessian = (hessian + hessian.T) / 2
    ref_cov = np.linalg.pinv(hessian)
    objective_at_fit = objective(coefficient)
    stationarity = float(np.max(np.abs(gradient(coefficient))))
    covariance_delta = float(np.max(np.abs(model.covariance_ - ref_cov)))
    objective_delta = objective_at_fit - best["objective"]
    normalized_value_delta = objective_at_fit + model.result_.log_likelihood
    reference_condition = float(np.linalg.cond(hessian))
    match = (
        model.result_.coefficient_converged
        and reference_condition < 1e10
        and np.linalg.eigvalsh(hessian)[0] > 0
        and best["gradient_inf"] < 1e-4
        and stationarity < 2e-4
        and abs(objective_delta) <= 2e-6 * max(1, abs(best["objective"]))
        and abs(normalized_value_delta) <= 2e-8 * max(1, abs(objective_at_fit))
        and covariance_delta <= 3e-3 * max(1, float(np.max(np.abs(ref_cov))))
    )
    return dict(
        status="matched" if match else "unresolved",
        scope="independent 64-row intercept law fit, not representative timing",
        case=asdict(small),
        identity=data["identity"],
        production=fitted,
        starts=endpoints,
        objective_delta=objective_delta,
        normalized_value_delta=normalized_value_delta,
        gradient_inf=stationarity,
        covariance_max_absolute=covariance_delta,
        reference_condition=reference_condition,
        covariance_reference=save_arrays(
            output.with_name(output.stem + "-oracle.npz"),
            {"covariance": ref_cov, "hessian": hessian},
        ),
        envelope="ordinary positive shapes; central float64 differences; Tweedie rate<=30/240 terms; NB finite endpoints<128/w",
        boundary="No high-precision or production-probability dependency; ill-conditioned/nonstationary endpoints stay unresolved",
    )


def worker(job_path):
    """One process per job; no worker spawns or waits on another fit."""
    job = json.loads(Path(job_path).read_text())
    output = Path(job["output"])
    record = dict(job, status="running", observations=[])
    write_json(output, record)
    started = time.perf_counter()
    try:
        sys.path.insert(0, job["source"]["path"])
        from threadpoolctl import threadpool_limits

        import superglm

        imported = Path(superglm.__file__).resolve().parent.parent
        if imported != Path(job["source"]["path"]).resolve():
            raise ValueError(f"wrong imported source: {imported}")
        actual = source_identity(
            imported, expected_sha=job["source"]["sha"], require_clean=job["role"] == "baseline"
        )
        if actual["digest"] != job["source"]["digest"]:
            raise ValueError("source content changed after scheduling this worker")
        if file_hash(__file__) != job["harness_sha256"]:
            raise ValueError("harness changed after scheduling this worker")
        record["source"] = actual
        case, config = Case(**job["case_spec"]), job["config"]
        with threadpool_limits(limits=1):
            environment, machine, threads = fingerprints()
            record.update(
                environment=environment,
                machine=machine,
                threads=threads,
                import_seconds=time.perf_counter() - started,
            )
            data = make_data(case, job["rows"], grid=config["grid"])
            spec = mathematical_spec(case, config)
            spec["family_config"] = family_for(case).to_config()
            record.update(
                spec=spec,
                identity=dict(
                    data["identity"],
                    spec=digest(spec),
                    environment=digest(environment),
                    machine=digest(machine),
                    threads=digest(threads),
                    harness=file_hash(__file__),
                ),
            )
            record["execution"] = {
                k: config[k] for k in ("discrete", "n_bins", "compact", "grid", "initial_lambda")
            }
            record["input_artifact"] = save_arrays(
                output.with_name(output.stem + "-inputs.npz"),
                {k: v for k, v in data.items() if hasattr(v, "dtype")},
            )
            record["adapter"] = offset_adapter(case, job["role"])
            write_json(output, record)
            mode = job["mode"]
            kinds = ("first", "warm") if mode == "timing" else (mode,)
            for kind in kinds:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    if mode == "reference":
                        record["reference"] = small_reference(
                            case, config, output.with_name(output.stem + "-reference")
                        )
                        record["reference"]["warnings"] = [str(w.message) for w in caught]
                        continue
                    profiler = cProfile.Profile() if mode == "profile" else None
                    if profiler:
                        profiler.enable()
                    clock = time.perf_counter()
                    try:
                        model, frame = complete_fit(case, data, config, job["role"])
                    except Exception:
                        record["failed_attempt"] = dict(
                            kind=kind,
                            elapsed_seconds=time.perf_counter() - clock,
                            warnings=[str(w.message) for w in caught],
                        )
                        raise
                    finally:
                        elapsed = time.perf_counter() - clock
                        if profiler:
                            profiler.disable()
                    # Write completion before potentially refusing public inference.
                    checkpoint = dict(
                        kind=kind,
                        status="fit_returned",
                        seconds=elapsed if mode == "timing" else None,
                    )
                    record["observations"].append(checkpoint)
                    write_json(output, record)
                    details = collect_fit(
                        model,
                        frame,
                        case,
                        data,
                        job["role"],
                        output.with_name(output.stem + f"-{kind}.npz"),
                        memory=mode == "memory",
                    )
                    checkpoint.update(
                        details, warnings=[*details["warnings"], *(str(w.message) for w in caught)]
                    )
                    if profiler:
                        checkpoint["profile"] = profile_summary(profiler)
                        if job["save_profile"]:
                            profile_path = output.with_suffix(".pstats")
                            profiler.dump_stats(profile_path)
                            checkpoint["profile_artifact"] = {
                                "path": str(profile_path),
                                "sha256": file_hash(profile_path),
                            }
                    write_json(output, record)
                    if mode == "memory":
                        with warnings.catch_warnings(record=True) as optional_warnings:
                            warnings.simplefilter("always")
                            try:
                                draws = model.posterior_draws(4, covariance="corrected", seed=62119)
                                checkpoint["optional_covariance"] = dict(
                                    requested="corrected",
                                    resolved=draws.covariance_kind,
                                    status="returned",
                                    scope="separate untimed memory job; no acceptance of unavailable correction",
                                )
                            except Exception as exc:
                                checkpoint["optional_covariance"] = dict(
                                    requested="corrected",
                                    status="refused",
                                    error=f"{type(exc).__name__}: {exc}",
                                )
                            checkpoint["optional_covariance"]["warnings"] = [
                                str(w.message) for w in optional_warnings
                            ]
                        write_json(output, record)
                    del model, frame
                    gc.collect()
            if data_identity(data) != data["identity"]:
                raise ValueError("fit mutated the synthetic input dataset or weights")
            if (
                source_identity(imported)["digest"] != actual["digest"]
                or file_hash(__file__) != job["harness_sha256"]
            ):
                raise ValueError("source/harness changed during this worker")
            record["status"] = "completed"
    except Exception as exc:
        status = (
            "missing_stage"
            if isinstance(exc, MissingStageError)
            else "refused"
            if type(exc).__name__ == "UnsupportedLikelihoodContractError"
            else "failed"
        )
        record.update(
            status=status, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()
        )
    record["worker_seconds"] = time.perf_counter() - started
    write_json(output, record)
    return 0 if record["status"] == "completed" else 1


def compact_run(record, path):
    kept = (
        "id",
        "case",
        "role",
        "repeat",
        "mode",
        "rows",
        "status",
        "error",
        "returncode",
        "log",
        "process_seconds",
        "worker_seconds",
        "import_seconds",
        "identity",
        "execution",
        "adapter",
        "observations",
        "reference",
        "input_artifact",
        "failed_attempt",
    )
    return {
        **{k: record[k] for k in kept if k in record},
        "receipt": str(path),
        "source": {k: record.get("source", {}).get(k) for k in ("path", "sha", "digest", "dirty")},
    }


def run_harness(args):
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"output already exists; choose a new receipt name: {output}")
    sources = {"candidate": source_identity(ROOT / "src")}
    if args.baseline_source:
        sources["baseline"] = source_identity(
            args.baseline_source, expected_sha=BASELINE, require_clean=True
        )
    selected = args.case or list(CASES)
    selected = [name for i, name in enumerate(selected) if i % args.shard_count == args.shard_index]
    if not selected:
        raise ValueError("empty shard")
    session = uuid.uuid4().hex
    artifacts = (ROOT / ".benchmark-artifacts" / (output.stem + "-" + session[:8])).resolve()
    artifacts.mkdir(parents=True)
    config = dict(
        discrete=args.discrete,
        compact=args.compact,
        n_bins=args.n_bins,
        grid=args.grid,
        initial_lambda=1.0,
        max_inner_iter=args.max_inner_iter,
        max_reml_iter=args.max_reml_iter,
    )
    receipt = dict(
        schema=SCHEMA,
        session=session,
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        harness_sha256=file_hash(__file__),
        baseline_expected=BASELINE,
        sources=sources,
        settings=vars(args),
        selected_cases={name: asdict(CASES[name]) for name in selected},
        runs=[],
        comparisons=[],
        absolute_cases=[],
        open_concerns=[
            "Task7A short evidence retains GG/GPD roughly 25%/20% timing increases; representative assessment still required",
            "Two-piece bounded mean-transform extra cost remains open",
            "Tweedie work limits/short timing concerns: keep all failed/refused/timed-out/nonconverged 20k/100k cells",
            "Task9 declared offsets and Task6 public Dirichlet lifecycle must be integrated before final validation",
            "No full-suite, native-platform, supported-version or representative-performance acceptance is inferred",
        ],
        measurement_policy={
            "order": "sequential baseline-before, candidate, baseline-after per case/repeat; no overlap",
            "first": "first complete fit in a fresh process and private empty NUMBA_CACHE_DIR; imports/data generation excluded",
            "warm": "new model complete fit in the same process after first-fit observation and model collection",
            "memory_profile": "separate fresh processes; never used as wall-time controls",
            "control_drift": "baseline-after / baseline-before; comparisons retained even when inadmissible",
            "artifacts": str(artifacts),
            "scope": "synthetic public data only",
        },
    )

    def checkpoint():
        write_json(output, receipt)

    def run(name, role, mode, repeat, label, settings=config):
        index = len(receipt["runs"])
        identifier = f"{index:04d}-{name}-{label}-{mode}"
        target = artifacts / (identifier + ".json")
        job = dict(
            id=identifier,
            case=name,
            case_spec=asdict(CASES[name]),
            role=role,
            mode=mode,
            repeat=repeat,
            rows=args.rows,
            source=sources[role],
            config=settings,
            output=str(target),
            save_profile=args.profile,
            session=session,
            harness_sha256=receipt["harness_sha256"],
        )
        job_path = artifacts / (identifier + "-job.json")
        write_json(job_path, job)
        env = os.environ.copy()
        env.update({key: "1" for key in THREAD_ENV})
        env.update(
            PYTHONPATH=sources[role]["path"],
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONHASHSEED="0",
            NUMBA_CACHE_DIR=str(artifacts / (identifier + "-jit")),
        )
        result = run_process(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(job_path)],
            target,
            timeout=args.timeout,
            env=env,
            context=job,
        )
        receipt["runs"].append(compact_run(result, target))
        checkpoint()
        print(f"{identifier}: {result['status']}", flush=True)
        return result

    def comparison(a, b, label, *, kind="warm", control_kind=None, execution_check=True):
        result = compare_records(
            a, b, kind=kind, control_kind=control_kind, execution_check=execution_check
        )
        receipt["comparisons"].append(
            dict(
                case=a["case"],
                candidate=a["id"],
                control=b["id"],
                comparison=label,
                kind=kind,
                **result,
            )
        )
        checkpoint()

    checkpoint()
    phases = args.phases.split(",")
    for name in selected:
        case = CASES[name]
        paired = case.baseline_supported and "baseline" in sources
        if not paired:
            receipt["absolute_cases"].append(
                {
                    "case": name,
                    "reason": "no same-law baseline exists"
                    if not case.baseline_supported
                    else "--baseline-source was not supplied",
                }
            )
        timed = {}
        if "timing" in phases:
            for repeat in range(args.repeats):
                before = run(name, "baseline", "timing", repeat, "before") if paired else None
                candidate = run(name, "candidate", "timing", repeat, "candidate")
                timed.setdefault("candidate", candidate)
                # Cold/warm equivalence is checked, not assumed from the warm endpoint.
                comparison(
                    candidate,
                    candidate,
                    "first/warm numerical identity",
                    kind="warm",
                    control_kind="first",
                )
                if paired:
                    after = run(name, "baseline", "timing", repeat, "after")
                    timed.setdefault("baseline", before)
                    for kind in ("first", "warm"):
                        comparison(candidate, before, "candidate/baseline-before", kind=kind)
                        comparison(candidate, after, "candidate/baseline-after", kind=kind)
                        comparison(after, before, "same-session control drift", kind=kind)
        for role in ("candidate", "baseline") if paired else ("candidate",):
            for mode in ("memory", "profile"):
                if mode in phases:
                    audit = run(name, role, mode, 0, role)
                    if role in timed:
                        comparison(
                            audit,
                            timed[role],
                            f"untimed {mode}/timing numerical identity",
                            kind=mode,
                            control_kind="warm",
                        )
        if "reference" in phases and (case.law in NEW_SCALARS or case.law == "dirichlet"):
            run(name, "candidate", "reference", 0, "independent-small")
        if args.equivalence:
            # Exact repeats and continuous approximation have distinct receipts;
            # no cross-grid equivalence or time ratio is asserted.
            for grid in ("repeat", "continuous"):
                dense = run(
                    name,
                    "candidate",
                    "validation",
                    0,
                    f"{grid}-dense",
                    dict(config, grid=grid, discrete=False),
                )
                discrete = run(
                    name,
                    "candidate",
                    "validation",
                    0,
                    f"{grid}-discrete",
                    dict(config, grid=grid, discrete=True),
                )
                comparison(
                    discrete,
                    dense,
                    "exact-repeat equivalence"
                    if grid == "repeat"
                    else "continuous-grid approximation sensitivity",
                    kind="validation",
                    execution_check=False,
                )
        if args.multiple_starts and case.fit == "smooth":
            starts = []
            for label, value in (("low", 0.1), ("ordinary", 1.0), ("high", 10.0)):
                starts.append(
                    run(
                        name,
                        "candidate",
                        "validation",
                        0,
                        f"start-{label}",
                        dict(config, initial_lambda=value),
                    )
                )
            for alternative in (starts[0], starts[2]):
                comparison(
                    alternative,
                    starts[1],
                    "smoothing start sensitivity (all endpoints retained)",
                    kind="validation",
                    execution_check=False,
                )
    receipt["status_counts"] = {
        status: sum(r["status"] == status for r in receipt["runs"])
        for status in sorted({r["status"] for r in receipt["runs"]})
    }
    receipt["validation_complete"] = False  # only parent owns the final acceptance receipt
    bad_runs = any(
        r["status"] != "completed"
        or any(
            o.get("public_status") != "completed"
            or not o.get("converged")
            or o.get("smoothing_converged") is False
            for o in r.get("observations", [])
        )
        or r.get("reference", {}).get("status") == "unresolved"
        for r in receipt["runs"]
    )
    bad_comparisons = any(
        not c["comparable"]
        for c in receipt["comparisons"]
        if c["comparison"]
        not in (
            "continuous-grid approximation sensitivity",
            "smoothing start sensitivity (all endpoints retained)",
        )
    )
    receipt["runner_status"] = "attention_required" if bad_runs or bad_comparisons else "executed"
    checkpoint()
    return int(bad_runs or bad_comparisons)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--case", action="append", choices=tuple(CASES))
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument(
        "--repeats", type=int, default=1, help="fresh-process baseline/candidate/control triplets"
    )
    parser.add_argument("--discrete", action="store_true")
    parser.add_argument(
        "--compact",
        action="store_true",
        help="fit with retain_rows=False; artifacts are always external",
    )
    parser.add_argument(
        "--profile", action="store_true", help="save full pstats from separate profiling fits"
    )
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument(
        "--timeout",
        type=float,
        default=180,
        help="seconds per worker, including its first/warm fits; partial failures retained",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--n-bins", type=int, default=256)
    parser.add_argument("--grid", choices=("repeat", "continuous"), default="repeat")
    parser.add_argument(
        "--equivalence",
        action="store_true",
        help="extra untimed dense/discrete fits on repeat and continuous grids",
    )
    parser.add_argument(
        "--multiple-starts", action="store_true", help="extra untimed REML starts 0.1, 1, 10"
    )
    parser.add_argument("--max-inner-iter", type=int, default=100)
    parser.add_argument("--max-reml-iter", type=int, default=100)
    parser.add_argument(
        "--phases",
        default="timing,memory,profile,reference",
        help="comma-separated timing,memory,profile,reference; partial runs remain partial",
    )
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args.worker)
    if args.list_cases:
        for name, case in CASES.items():
            print(
                f"{name}\t{canonical(asdict(case))}\tbaseline={'same-law' if case.baseline_supported else 'unavailable'}"
            )
        return 0
    if args.output is None:
        parser.error("--output is required for a run")
    if (
        min(
            args.rows,
            args.repeats,
            args.shard_count,
            args.n_bins,
            args.max_inner_iter,
            args.max_reml_iter,
        )
        < 1
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
    ):
        parser.error(
            "rows, repeats, timeouts, bins, iteration limits and shard count must be positive"
        )
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, --shard-count)")
    if not args.phases or not set(args.phases.split(",")) <= {
        "timing",
        "memory",
        "profile",
        "reference",
    }:
        parser.error("unknown or empty --phases")
    try:
        return run_harness(args)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
