"""Fit records for benchboard: one worker-protocol-v1 spool file per fit.

Imported only by ``superglm._fit_gate`` and only while ``BENCHBOARD_FIT_DIR``
is set, which benchboard does for the jobs it runs.  Every function here is
best effort: a failure drops the record (or one of its fields) and never
reaches the fit.  Rows, responses and coefficients are summarised, never
written.  The payload has the shape ``benchmarks/books.py`` writes: a
``config`` of what was known before the fit, a ``summary`` of outcomes, the
fit's phases and the full record as diagnostics.
"""

from __future__ import annotations

import contextvars
import gzip
import json
import math
import os
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path

PROTOCOL = 1
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "SUPERGLM_BLAS_THREADS",
    "SUPERGLM_N_JOBS",
)
_parent = contextvars.ContextVar("superglm_fit_record_parent", default=None)
_process: dict = {}
_sequence = [0]
last_fit_id: str | None = None


def _plain(value, depth=0):
    """JSON-safe copy of scalars and small containers; arrays become their shape."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        if value.shape == ():
            return _plain(value.item(), depth)
        return {"array": list(value.shape), "dtype": str(value.dtype)}
    if depth > 6:
        return type(value).__name__
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(k): _plain(v, depth + 1) for k, v in dict(value).items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(v, depth + 1) for v in value][:200]
    return type(value).__name__


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001 - a missing field never fails a record
        return {"unavailable": f"{type(exc).__name__}: {exc}"[:200]}


def _steal_ticks():
    try:
        fields = [int(v) for v in Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:]]
        return sum(fields), fields[7] if len(fields) > 7 else 0
    except (OSError, ValueError, IndexError):
        return None


def _load():
    try:
        return list(os.getloadavg())
    except (AttributeError, OSError):
        return None


def _peak_rss_mib():
    import resource  # POSIX only; elsewhere the field says why it is missing

    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 2**20 if sys.platform == "darwin" else raw / 1024.0


def _git(root, *args):
    out = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=5, check=True
    )
    return out.stdout.strip()


def _process_facts() -> dict:
    """Per-process facts: the commit superglm was imported from, packages and machine."""
    if _process.get("pid") == os.getpid():
        return _process
    import importlib.metadata

    root = Path(__file__).resolve().parents[1]
    commit = {"sha": None, "dirty": None}
    try:
        top = _git(root, "rev-parse", "--show-toplevel")
        commit = {
            "sha": _git(top, "rev-parse", "HEAD"),
            "dirty": bool(_git(top, "status", "--porcelain", "--", "src")),
        }
    except Exception:  # noqa: BLE001 - an installed wheel has no git tree
        pass
    packages = {}
    for name in ("superglm", "numpy", "scipy", "numba", "pandas", "tabmat", "threadpoolctl"):
        try:
            packages[name] = importlib.metadata.version(name)
        except Exception:  # noqa: BLE001
            packages[name] = None
    cpu, cores, ram = platform.processor() or None, None, None
    try:
        physical, phys = set(), None
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            key, _, value = (part.strip() for part in line.partition(":"))
            if key == "model name":
                cpu = value
            elif key == "physical id":
                phys = value
            elif key == "core id":
                physical.add((phys, value))
        cores = len(physical) or None
        ram = next(
            round(int(line.split()[1]) / 2**20, 1)
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemTotal:")
        )
    except (OSError, StopIteration, ValueError):
        pass
    _process.clear()
    _process.update(
        pid=os.getpid(),
        commit=commit,
        environment={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": packages,
        },
        machine={
            "host": platform.node(),
            "cpu_model": cpu,
            "physical_cores": cores,
            "logical_cpus": os.cpu_count(),
            "ram_gib": ram,
        },
    )
    return _process


def _threads() -> dict:
    """Thread pools as this process sees them, not as the environment asked."""
    out = {"env": {k: os.environ.get(k) for k in _THREAD_VARS}}
    try:
        from threadpoolctl import threadpool_info

        out["pools"] = [
            {
                "api": i.get("internal_api"),
                "threads": i.get("num_threads"),
                "version": i.get("version"),
                "architecture": i.get("architecture"),
            }
            for i in threadpool_info()
        ]
    except Exception as exc:  # noqa: BLE001
        out["pools"] = f"unavailable: {exc}"
    numba = sys.modules.get("numba")
    out["numba"] = getattr(getattr(numba, "config", None), "NUMBA_NUM_THREADS", None)
    return out


# --------------------------------------------------------------------------- descriptors


def _vector_stats(v) -> dict | None:
    import numpy as np

    if v is None:
        return None
    a = np.asarray(v, dtype=np.float64).ravel()
    if a.size == 0:
        return {"n": 0}
    mean = float(a.mean())
    return {
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": mean,
        "cv": float(a.std() / mean) if mean else None,
        "share_zero": float(np.mean(a == 0)),
    }


def _gm_facts(gm) -> dict:
    """Width, cardinality, basis non-zeros per row and empty cells of one group matrix."""
    import numpy as np

    n = gm.shape[0]
    out = {"type": type(gm).__name__, "width": int(gm.shape[1])}
    codes = getattr(gm, "codes", None)
    if codes is not None and hasattr(gm, "n_levels"):
        L = int(gm.n_levels)
        counts = np.bincount(np.asarray(codes), minlength=L + 1)
        out.update(
            cardinality=L + 1,
            nnz_per_row=float(counts[:L].sum() / n),
            empty_cells=int(np.count_nonzero(counts[: L + 1] == 0)),
            min_level_count=int(counts[: L + 1].min()),
        )
        return out
    B = getattr(gm, "B", None)
    if B is not None and hasattr(B, "nnz"):
        out.update(nnz_per_row=float(B.nnz / n), basis_width=int(B.shape[1]))
        return out
    if hasattr(gm, "B_unique") and hasattr(gm, "bin_idx"):
        per_bin = np.count_nonzero(gm.B_unique, axis=1)
        out.update(
            cardinality=int(gm.n_bins), nnz_per_row=float(per_bin[np.asarray(gm.bin_idx)].mean())
        )
        return out
    if hasattr(gm, "B1_unique_t") and hasattr(gm, "idx1"):
        nz1 = np.count_nonzero(gm.B1_unique_t, axis=1)[np.asarray(gm.idx1)]
        nz2 = np.count_nonzero(gm.B2_unique_t, axis=1)[np.asarray(gm.idx2)]
        cells = int(gm.n_bins1) * int(gm.n_bins2)
        joint = np.unique(
            np.asarray(gm.idx1, dtype=np.int64) * int(gm.n_bins2) + np.asarray(gm.idx2)
        ).size
        out.update(
            cardinality=joint, nnz_per_row=float(np.mean(nz1 * nz2)), empty_cells=cells - joint
        )
        return out
    if hasattr(gm, "B_unique") and hasattr(gm, "row_idx") and hasattr(gm, "bin_idx_level"):
        per_bin = np.count_nonzero(gm.B_unique, axis=1)
        out.update(
            cardinality=int(gm.n_bins),
            nnz_per_row=float(per_bin[np.asarray(gm.bin_idx_level)].sum() / n),
        )
        return out
    if type(gm).__name__.startswith("Dense"):
        out["nnz_per_row"] = float(gm.shape[1])
    return out


def _terms(groups, design, kinds) -> list[dict]:
    """Per term (feature or pair): kind, width, cardinality, nnz per row, empty cells."""
    terms: dict[str, dict] = {}
    for group, gm in zip(groups, design.group_matrices, strict=False):
        name = getattr(group, "feature_name", None) or group.name
        facts = _safe(_gm_facts, gm)
        term = terms.setdefault(
            name,
            {
                "name": name,
                "kind": kinds.get(name),
                "pair": ":" in name,
                "width": 0,
                "groups": [],
                "nnz_per_row": 0.0,
            },
        )
        term["width"] += int(getattr(group, "size", 0) or 0)
        term["groups"].append(facts)
        if isinstance(facts.get("nnz_per_row"), float) and term["nnz_per_row"] is not None:
            term["nnz_per_row"] += facts["nnz_per_row"]
        else:
            term["nnz_per_row"] = None
        for key, combine in (("cardinality", max), ("empty_cells", sum), ("min_level_count", min)):
            if isinstance(facts.get(key), int):
                term[key] = combine((term[key], facts[key])) if key in term else facts[key]
    return list(terms.values())


def _design_summary(terms, n_rows) -> dict:
    nnz = [t["nnz_per_row"] for t in terms]
    return {
        "terms": sum(not t["pair"] for t in terms),
        "pairs": sum(t["pair"] for t in terms),
        "groups": sum(len(t["groups"]) for t in terms),
        "nnz_per_row": float(sum(nnz)) if all(v is not None for v in nnz) else None,
        "empty_cells": sum(t.get("empty_cells", 0) for t in terms),
        "min_level_count": min(
            (t["min_level_count"] for t in terms if "min_level_count" in t), default=None
        ),
        "max_cardinality": max((t.get("cardinality", 0) for t in terms), default=None),
    }


_INPUTS = ("self", "X", "y", "sample_weight", "offset", "offsets")


def _options(params) -> dict:
    return _plain({k: v for k, v in params.items() if k not in _INPUTS and not k.startswith("_")})


def _data(params, lss) -> dict:
    X, y, w = params.get("X"), params.get("y"), params.get("sample_weight")
    off = params.get("offsets" if lss else "offset")
    out = {
        "n_rows": int(len(y)) if y is not None else None,
        "n_input_columns": len(getattr(X, "columns", ())),
    }
    out["response"] = _safe(_vector_stats, y)
    out["weights"] = _safe(_vector_stats, w)
    out["offset_present"] = off is not None
    if off is not None and not lss:
        import numpy as np

        out["exposure_from_offset"] = _safe(
            _vector_stats, np.exp(np.asarray(off, dtype=np.float64))
        )
    elif off is not None:
        out["offset_predictors"] = sorted(off)
    return out


# --------------------------------------------------------------------------- the model


def _glm(model, params) -> dict:
    cfg = model._config
    family = cfg.family if isinstance(cfg.family, str) else type(cfg.family).__name__
    link = cfg.link if isinstance(cfg.link, str) or cfg.link is None else type(cfg.link).__name__
    kinds = {
        k: type(v).__name__
        for k, v in {
            **getattr(model, "_specs", {}),
            **getattr(model, "_interaction_specs", {}),
        }.items()
    }
    out = {
        "model": "SuperGLM",
        "distribution": str(family).lower(),
        "link": link,
        "discrete": bool(cfg.discrete),
        "n_bins": _plain(cfg.n_bins),
        "direct_solve": _plain(cfg.direct_solve),
        "options": _options(params),
    }
    dm = getattr(model, "_dm", None)
    if dm is not None:
        terms = _terms(model._groups, dm, kinds)
        out.update(n_columns=int(dm.p), term_list=terms, design=_design_summary(terms, dm.p))
    return out


def _glm_outcome(model) -> dict:
    profile = getattr(model, "_reml_profile", None) or {}
    scalar = {
        k: _plain(v)
        for k, v in profile.items()
        if isinstance(v, (int, float, str, bool)) or v is None
    }
    times = {
        k: v for k, v in scalar.items() if isinstance(v, float) and (k.endswith("_s") or "_s_" in k)
    }
    out = {
        "phases": times,
        "phase_source": "reml_profile (nested, not exclusive)",
        "work": {
            k: v
            for k, v in scalar.items()
            if k not in times and isinstance(v, (int, float)) and not isinstance(v, bool)
        },
        "flags": {k: v for k, v in scalar.items() if isinstance(v, (bool, str))},
    }
    result = getattr(model, "_result", None)
    if result is not None:
        out["n_iter"] = _plain(getattr(result, "n_iter", None))
        out["converged"] = _plain(getattr(result, "converged", None))
        out["edf"] = _plain(getattr(result, "effective_df", None))
        out["solver"] = _plain(getattr(result, "direct_backend", None))
    if profile:
        diag = _safe(model.reml_diagnostics)
        if "unavailable" not in diag:
            out.update(
                n_iter=_plain(diag.get("n_reml_iter")),
                converged=_plain(diag.get("converged")),
                termination_reason=_plain(diag.get("termination_reason")),
                objective=_plain(diag.get("objective")),
                n_lambdas=len(diag.get("lambdas") or {}),
            )
    out["certification"] = {k: v for k, v in out["flags"].items() if "certif" in k}
    return out


def _lss(model, params) -> dict:
    family = type(model._family).__name__
    out = {
        "model": "SuperLSS",
        "distribution": family,
        "link": None,
        "discrete": bool(model._discrete),
        "n_bins": _plain(model._n_bins),
        "options": _options(params),
        "predictors": [
            {
                "name": p.name,
                "features": {k: type(v).__name__ for k, v in p.features.items()},
                "interactions": [list(i) for i in p.interactions],
            }
            for p in model._predictors
        ],
    }
    return out


def _lss_outcome(model) -> dict:
    out = {}
    fitted = getattr(model, "_model", None)
    if fitted is None:
        return out
    state = fitted.fit_state
    terms = []
    for p in state.layout.predictors:
        kinds = {}
        terms += [dict(t, predictor=p.name) for t in _terms(p.groups, p.design, kinds)]
    out.update(
        n_columns=int(state.layout.n_coefficients),
        term_list=terms,
        design=_design_summary(terms, None),
    )
    snapshot = getattr(model, "_fit_phase_snapshot", None)
    if snapshot is not None:
        snap = snapshot.as_dict()
        out.update(
            phases=dict(snap.get("exclusive_seconds") or {}),
            phase_source="exclusive",
            work={f"count:{k}": v for k, v in (snap.get("counts") or {}).items()},
        )
    sm = state.smoothing
    out.update(
        converged=_plain(getattr(sm, "converged", None)),
        termination_reason=str(getattr(sm, "convergence_reason", None)),
        n_iter=_plain(getattr(sm, "iterations", None)),
        objective=_plain(getattr(sm, "objective", None)),
        edf=_plain(sum(state.result.predictor_edf.values())),
    )
    profile = _safe(lambda: model.diagnose().profile)
    if not isinstance(profile, dict):
        out.setdefault("work", {}).update(
            outer_iterations=profile.outer_iterations,
            coefficient_fits=profile.coefficient_fits,
            inner_iterations=profile.inner_iterations,
        )
    return out


# --------------------------------------------------------------------------- begin / end


def begin(model, method, params) -> dict:
    """Facts taken before the fit; the clock starts last."""
    lss = type(model).__name__ == "SuperLSS"
    token = {
        "id": uuid.uuid4().hex,
        "method": method,
        "lss": lss,
        "parent": _parent.get(),
        "params": dict(params),
        "load_start": _load(),
        "ticks_start": _steal_ticks(),
        "threads": _safe(_threads),
        "data": _safe(_data, params, lss),
    }
    token["reset"] = _parent.set(token["id"])
    token["started_at"], token["t0"] = time.time(), time.perf_counter()
    return token


def end(token, model, error) -> None:
    """Write the record; never raises."""
    global last_fit_id
    wall = time.perf_counter() - token["t0"]
    ended = time.time()
    try:
        _parent.reset(token["reset"])
    except Exception:  # noqa: BLE001
        pass
    try:
        spool, task = os.environ.get("BENCHBOARD_FIT_DIR"), os.environ.get("BENCHBOARD_TASK_ID")
        if not spool or not task:
            return
        facts = _process_facts()
        before = _safe(_lss if token["lss"] else _glm, model, token["params"])
        outcome = (
            {}
            if error is not None
            else _safe(_lss_outcome if token["lss"] else _glm_outcome, model)
        )
        if token["lss"] and "n_columns" in outcome:
            before.update({k: outcome.pop(k) for k in ("n_columns", "term_list", "design")})
        if error is not None:  # the design on the model may be an earlier fit's
            for key in ("n_columns", "term_list", "design"):
                before.pop(key, None)
        ticks = _steal_ticks()
        steal = None
        if ticks and token["ticks_start"]:
            total = ticks[0] - token["ticks_start"][0]
            steal = 100.0 * (ticks[1] - token["ticks_start"][1]) / total if total > 0 else None
        data = token["data"] if isinstance(token["data"], dict) else {}
        design = before.get("design") or {}
        config = {
            "n_rows": data.get("n_rows"),
            "n_features": design.get("terms"),
            "n_pairs": design.get("pairs"),
            "distribution": before.get("distribution"),
            "link": before.get("link"),
            "model": before.get("model"),
            "discrete": before.get("discrete"),
            "options": before.get("options"),
            "data": data,
            "design": design,
            "terms": before.get("term_list"),
            "predictors": before.get("predictors"),
            "commit": facts["commit"],
            "threads": token["threads"],
            "load_start": token["load_start"],
        }
        summary = {
            "n_columns": before.get("n_columns"),
            "solver": outcome.get("solver"),
            "converged": outcome.get("converged"),
            "n_iter": outcome.get("n_iter"),
            "termination_reason": outcome.get("termination_reason"),
            "objective": outcome.get("objective"),
            "edf": outcome.get("edf"),
            "certification": outcome.get("certification"),
            "work": outcome.get("work"),
            "peak_rss_mib": _safe(_peak_rss_mib),
            "load_end": _load(),
            "steal_percent": steal,
            "error": None if error is None else f"{type(error).__name__}: {error}"[:300],
        }
        phases = [
            {"name": k, "wall_s": float(v), "source": outcome.get("phase_source")}
            for k, v in (outcome.get("phases") or {}).items()
            if isinstance(v, (int, float)) and math.isfinite(v) and v >= 0
        ]
        _sequence[0] += 1
        fit = {
            "id": token["id"],
            "parent_fit_id": token["parent"],
            "sequence": _sequence[0],
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "method": token["method"],
            "status": "success"
            if error is None
            else ("interrupted" if isinstance(error, KeyboardInterrupt) else "failed"),
            "started_at": token["started_at"],
            "ended_at": ended,
            "wall_s": wall,
            "config": _plain(config),
            "summary": _plain(summary),
            "phases": phases,
            "diagnostics": _plain(
                {"flags": outcome.get("flags"), "phases_all": outcome.get("phases")}
            ),
        }
        payload = {
            "protocol": PROTOCOL,
            "task_id": task,
            "environment": _plain({**facts["environment"], "machine": facts["machine"]}),
            "fits": [fit],
        }
        target = Path(spool) / f"superglm-{os.getpid()}-{token['id']}.json.gz"
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_bytes(
            gzip.compress(json.dumps(payload, allow_nan=False).encode(), compresslevel=1, mtime=0)
        )
        tmp.replace(target)
        last_fit_id = token["id"]
    except Exception:  # noqa: BLE001, S110 - recording never fails or slows a fit's caller beyond this
        pass
