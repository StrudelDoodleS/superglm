"""One evidence harness for complete fits on named books, on top of benchboard's baselines.

Subcommands (``python benchmarks/books.py <cmd> --help`` for options)::

    cases                       list the named case sets
    run CASE --threads T        one fit in this process: one JSON record, and the fit written
                                into benchboard's fit history (fits, fit_phases, diagnostics)
    baseline [REF] --set NAME   time a case set at REF in a detached worktree, one benchboard
                                run-timed per fit (--record for pinned runs), reusing any key
                                already stored
    compare [RECORD ...]        branch records against the stored baseline at the merge base

A record is keyed by (git SHA, dataset hash, case-spec hash, environment,
machine, thread setting).  benchboard keys a baseline by (case, commit,
hardware), so the benchboard case name carries a hash of everything but the
SHA.  A stored baseline with the same key is never re-timed, and ``compare``
refuses, naming the component, when any part of the key other than the SHA
differs.  See docs/development/benchmark-baselines.md.

Only the standard library is imported at module level: the thread setting has
to reach the environment before numpy, numba or any BLAS is loaded.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import threading
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS_ROOT = HERE.parent
SCHEMA = 1
DEFAULT_OUT = Path(
    os.environ.get("SUPERGLM_BOOKS_DIR", Path.home() / ".cache" / "benchboard" / "books")
)
TREES = Path(
    os.environ.get("SUPERGLM_BENCH_TREES", Path.home() / ".cache" / "superglm-bench" / "trees")
)
UPSTREAM = "origin/master"

# Every pool a fit can fan out on.  "pinned" sets each to 1; "default" removes them all.
POOLS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "RAYON_NUM_THREADS",
    "SUPERGLM_BLAS_THREADS",
    "SUPERGLM_N_JOBS",
)
THREADS = ("pinned", "default")

# --------------------------------------------------------------------------- cases

SEED = 20261012
WARMUP_GLM = {"rows": 3000, "max_reml_iter": 2}

FEATURES = {
    "DrivAge": ["spline", {"n_knots": 8}],
    "VehAge": ["spline", {"n_knots": 12}],
    "BonusMalus": [
        "spline",
        {"n_knots": 12, "knot_strategy": "quantile_tempered", "knot_alpha": 0.2},
    ],
    "LogDensity": ["numeric", {}],
    "VehBrand": ["categorical", {}],
    "Region": ["categorical", {}],
}
PAIRS4 = [
    ["DrivAge", "BonusMalus"],
    ["VehAge", "BonusMalus"],
    ["BonusMalus", "VehBrand"],
    ["BonusMalus", "Region"],
]
PAIRS10 = [
    ["VehAge", "VehBrand"],
    ["VehAge", "BonusMalus"],
    ["DrivAge", "BonusMalus"],
    ["BonusMalus", "VehBrand"],
    ["DrivAge", "Region"],
    ["DrivAge", "VehAge"],
    ["DrivAge", "VehBrand"],
    ["BonusMalus", "Region"],
    ["LogDensity", "VehBrand"],
    ["VehAge", "Region"],
]
# The four top-ranked screened pairs the family/link audit used.
PAIRS4_FAMILIES = [
    ["VehAge", "VehBrand"],
    ["VehAge", "BonusMalus"],
    ["DrivAge", "BonusMalus"],
    ["BonusMalus", "VehBrand"],
]

BOOK_STANDARD = {
    "book": "fremtpl2",
    "table": "freq",
    "exposure_floor": 0.01,
    "casts": [],
    "seed": SEED,
    "train": 0.8,
}
CASTS_FAMILIES = [
    ["DrivAge", "float"],
    ["VehAge", "float"],
    ["BonusMalus", "float"],
    ["VehBrand", "str"],
    ["Region", "str"],
]
CASTS_LSS = CASTS_FAMILIES + [["VehPower", "float"], ["VehGas", "str"], ["Area", "str"]]
BOOK_FAMILIES = dict(BOOK_STANDARD, exposure_floor=None, casts=CASTS_FAMILIES)
BOOK_LSS = dict(BOOK_STANDARD, exposure_floor=None, casts=CASTS_LSS)


def _standard(pairs, discrete):
    return {
        "kind": "glm",
        "dataset": BOOK_STANDARD,
        "family": ["poisson", {}],
        "link": None,
        "response": "claim_rate",
        "weights": "exposure",
        "offset": None,
        "model_options": {"weight_semantics": "frequency", "discrete": discrete, "n_bins": 256},
        "features": FEATURES,
        "interactions": pairs,
        "fit": ["fit_reml", {}],
        "warmup": WARMUP_GLM,
        "nll": "poisson_counts",
    }


# name -> (superglm.families constructor and kwargs, link, response, weights, offset, table)
_FAMILY_CASES = {
    "poisson_log": (["poisson", {}], "log", "claim_rate", "exposure", None, "freq"),
    "gamma_log": (["gamma", {}], "log", "amount", None, None, "sev"),
    "gamma_inverse": (["gamma", {}], "inverse", "amount_thousands", None, None, "sev"),
    "tweedie_fixed": (["tweedie", {"p": 1.5}], "log", "pure_premium", "exposure", None, "freq"),
    "tweedie_estp": (["tweedie", {"p": 1.5}], "log", "pure_premium", "exposure", None, "freq"),
    "nb_fixed": (["nb2", {"theta": 1.0}], "log", "claim_count", None, "log_exposure", "freq"),
    "nb_auto": (["nb2", {"theta": "auto"}], "log", "claim_count", None, "log_exposure", "freq"),
    "binomial_logit": (["binomial", {}], "logit", "any_claim", None, None, "freq"),
    "binomial_probit": (["binomial", {}], "probit", "any_claim", None, None, "freq"),
    "binomial_cloglog": (["binomial", {}], "cloglog", "any_claim", None, None, "freq"),
    "binomial_log": (["binomial", {}], "log", "any_claim", None, "log_exposure", "freq"),
    "gaussian_identity": (["gaussian", {}], "identity", "log_amount", None, None, "sev"),
    "gaussian_log": (["gaussian", {}], "log", "amount_thousands", None, None, "sev"),
}


def _family_case(name, discrete):
    family, link, response, weights, offset, table = _FAMILY_CASES[name]
    return {
        "kind": "glm",
        "dataset": dict(BOOK_FAMILIES, table=table),
        "family": family,
        "link": link,
        "response": response,
        "weights": weights,
        "offset": offset,
        "model_options": {"discrete": discrete},
        "features": FEATURES,
        "interactions": PAIRS4_FAMILIES,
        "fit": ["estimate_p", {"fit_mode": "reml"}] if name == "tweedie_estp" else ["fit_reml", {}],
        "warmup": WARMUP_GLM,
        "nll": "superglm_weighted",
    }


_LOC = [
    ["s", "DrivAge", {"kind": "cr", "k": 10}],
    ["s", "VehAge", {"kind": "cr", "k": 10}],
    ["s", "BonusMalus", {"kind": "cr", "k": 10}],
    ["s", "LogDensity", {"kind": "cr", "k": 8}],
    ["s", "VehPower", {"kind": "cr", "k": 6}],
    ["cat", "VehBrand", {}],
    ["cat", "Region", {}],
    ["cat", "VehGas", {}],
    ["cat", "Area", {}],
]
_SCALE = [
    ["s", "DrivAge", {"kind": "cr", "k": 8}],
    ["s", "VehAge", {"kind": "cr", "k": 8}],
    ["s", "BonusMalus", {"kind": "cr", "k": 8}],
    ["s", "LogDensity", {"kind": "cr", "k": 6}],
    ["cat", "VehGas", {}],
    ["re", "VehBrand", {}],
    ["re", "Region", {}],
]
_SMALL = [["s", "DrivAge", {"kind": "cr", "k": 5}], ["cat", "VehGas", {}], ["cat", "Area", {}]]


def _lss(family, predictors, response, weights, offsets, table, warmup_rows):
    return {
        "kind": "lss",
        "dataset": dict(BOOK_LSS, table=table),
        "family": [family, {}],
        "predictors": predictors,
        "response": response,
        "weights": weights,
        "offsets": offsets,
        "model_options": {"discrete": False},
        "fit": ["fit_reml", {}],
        "warmup": {"rows": warmup_rows, "max_reml_iter": 2},
    }


SYNTHETIC = {"book": "synthetic", "rows": 3000, "test_rows": 1000, "seed": 20261015}
SMOKE = {
    "kind": "glm",
    "dataset": SYNTHETIC,
    "family": ["poisson", {}],
    "link": None,
    "response": "claim_rate",
    "weights": "exposure",
    "offset": None,
    "model_options": {"weight_semantics": "frequency", "discrete": False},
    "features": {
        "a": ["spline", {"n_knots": 6}],
        "b": ["spline", {"n_knots": 6}],
        "c": ["categorical", {}],
        "d": ["numeric", {}],
    },
    "interactions": [["a", "c"]],
    "fit": ["fit_reml", {}],
    "warmup": {"rows": 0, "max_reml_iter": 2},
    "nll": "poisson_counts",
}

CASES = {
    "std_mains": _standard([], False),
    "std_pairs4": _standard(PAIRS4, False),
    "std_pairs10": _standard(PAIRS10, False),
    "std_pairs4_binned": _standard(PAIRS4, True),
    **{
        f"fam_{n}_{m}": _family_case(n, m == "binned")
        for n in _FAMILY_CASES
        for m in ("exact", "binned")
    },
    "lss_nb": _lss(
        "NegativeBinomialLS",
        [["mean", _LOC], ["theta", _SMALL]],
        "claim_count",
        None,
        {"mean": "log_exposure", "theta": "log_exposure"},
        "freq",
        60000,
    ),
    "lss_tweedie": _lss(
        "TweedieLSS",
        [["mu", _LOC], ["phi", _SMALL], ["p", []]],
        "pure_premium",
        "exposure",
        None,
        "freq",
        60000,
    ),
    "lss_gamma": _lss(
        "GammaLS", [["mean", _LOC], ["scale", _SCALE]], "amount", None, None, "sev", 4000
    ),
    "smoke_poisson": SMOKE,
}
SETS = {
    "standard": ["std_mains", "std_pairs4", "std_pairs10", "std_pairs4_binned"],
    "families": [c for c in CASES if c.startswith("fam_")],
    "lss": ["lss_nb", "lss_tweedie", "lss_gamma"],
    "smoke": ["smoke_poisson"],
}


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def digest(obj) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def spec_hash(case: str) -> str:
    return digest({"schema": SCHEMA, "case": case, "spec": CASES[case]})


# --------------------------------------------------------------------------- data

_FRAMES: dict = {}


def _fremtpl2():
    spec = importlib.util.spec_from_file_location(
        "_books_fremtpl2", HARNESS_ROOT / "examples" / "fremtpl2_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_book(dataset: dict):
    """Prepared (train, test) frames for a dataset spec; cached per process."""
    key = canonical(dataset)
    if key not in _FRAMES:
        _FRAMES[key] = (
            _synthetic(dataset) if dataset["book"] == "synthetic" else _fremtpl2_book(dataset)
        )
    return _FRAMES[key]


def _fremtpl2_book(d):
    import numpy as np
    import pandas as pd

    fd = _fremtpl2()
    freq = fd.load_corrected_freq()
    if d["exposure_floor"] is not None:
        freq["Exposure"] = freq["Exposure"].clip(lower=d["exposure_floor"])
    freq["LogDensity"] = np.log1p(freq["Density"].to_numpy(dtype=np.float64))
    for column, kind in d["casts"]:
        freq[column] = freq[column].astype(np.float64 if kind == "float" else str)
    perm = np.random.default_rng(d["seed"]).permutation(len(freq))
    is_train = np.zeros(len(freq), dtype=bool)
    is_train[perm[: int(round(d["train"] * len(freq)))]] = True
    freq["is_train"] = is_train
    frame = freq
    if d["table"] == "sev":
        sev = pd.read_parquet(fd.find_data_dir() / fd.SEV_FILE)
        cols = ["IDpol"] + [
            c for c in freq.columns if c not in sev.columns and c not in ("ClaimNb", "ClaimTotal")
        ]
        frame = sev.merge(freq[cols], on="IDpol", how="inner")
        frame = frame[frame["ClaimAmount"] > 0]
    train = frame[frame["is_train"]].reset_index(drop=True)
    test = frame[~frame["is_train"]].reset_index(drop=True)
    return train, test


def _synthetic(d):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(d["seed"])
    n = d["rows"] + d["test_rows"]
    frame = pd.DataFrame(
        {
            "a": np.round(rng.uniform(0, 1, n), 2),
            "b": np.round(rng.uniform(-2, 2, n), 2),
            "c": rng.integers(0, 6, n).astype(str),
            "d": rng.normal(10.0, 2.0, n),
        }
    )
    eta = (
        0.6 * np.sin(3 * frame.a)
        + 0.15 * frame.b**2
        - 1.0
        + 0.15 * (frame.c.astype(int) % 3)
        + 0.03 * (frame.d - 10)
    )
    frame["Exposure"] = rng.uniform(0.5, 1.5, n)
    frame["ClaimNb"] = rng.poisson(frame["Exposure"] * np.exp(eta)).astype(np.float64)
    return frame.iloc[: d["rows"]].reset_index(drop=True), frame.iloc[d["rows"] :].reset_index(
        drop=True
    )


def dataset_hash(dataset: dict) -> str:
    """Content hash of the prepared train and test frames, column names and dtypes included."""
    import pandas as pd

    h = hashlib.sha256()
    for frame in load_book(dataset):
        h.update(canonical([[str(c), str(t)] for c, t in frame.dtypes.items()]).encode())
        h.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return h.hexdigest()


def _response(frame, name):
    import numpy as np

    if name in ("claim_rate", "claim_count", "any_claim"):
        counts = frame["ClaimNb"].to_numpy(np.float64)
        if name == "claim_rate":
            return counts / frame["Exposure"].to_numpy(np.float64)
        return counts if name == "claim_count" else (counts > 0).astype(np.float64)
    if name == "pure_premium":
        return frame["ClaimTotal"].to_numpy(np.float64) / frame["Exposure"].to_numpy(np.float64)
    amount = frame["ClaimAmount"].to_numpy(np.float64)
    return {"amount": amount, "amount_thousands": amount / 1000.0, "log_amount": np.log(amount)}[
        name
    ]


def _column(frame, name):
    import numpy as np

    if name is None:
        return None
    expo = frame["Exposure"].to_numpy(np.float64)
    return {"exposure": expo, "log_exposure": np.log(expo)}[name]


# --------------------------------------------------------------------------- provenance


def _git(tree, *args) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(tree), *args], capture_output=True, text=True, timeout=60, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def tree_state(path: Path, paths=("src", "pyproject.toml", "uv.lock")) -> dict:
    """The commit a path belongs to; dirty when ``paths`` carry changes, untracked files included."""
    top = _git(path, "rev-parse", "--show-toplevel")
    if top is None:
        return {"tree": None, "sha": None, "dirty": None}
    status = _git(top, "status", "--porcelain", "--", *paths)
    return {"tree": top, "sha": _git(top, "rev-parse", "HEAD"), "dirty": bool(status)}


def environment() -> dict:
    import numba
    import numpy
    import pandas
    import scipy
    import scipy.linalg  # noqa: F401 - loads scipy's own BLAS before the pools are listed
    from threadpoolctl import threadpool_info

    blas = [
        {
            "user_api": i.get("user_api"),
            "internal_api": i.get("internal_api"),
            "version": i.get("version"),
            "architecture": i.get("architecture"),
            "threading_layer": i.get("threading_layer"),
            "library": Path(i.get("filepath", "")).name,
        }
        for i in threadpool_info()
    ]
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "numba": numba.__version__,
        "pandas": pandas.__version__,
        "blas": sorted(blas, key=canonical),
    }


def machine() -> dict:
    model, cores, ram = platform.processor() or None, None, None
    try:
        physical, phys_id = set(), None
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            if key == "model name":
                model = value
            elif key == "physical id":
                phys_id = value
            elif key == "core id":
                physical.add((phys_id, value))
        cores = len(physical) or None
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                ram = round(int(line.split()[1]) / 2**20, 1)
    except OSError:
        pass
    if cores is None or ram is None:
        try:
            import psutil

            cores = cores or psutil.cpu_count(logical=False)
            ram = ram or round(psutil.virtual_memory().total / 2**30, 1)
        except ImportError:
            pass
    return {
        "cpu_model": model,
        "physical_cores": cores,
        "logical_cpus": os.cpu_count(),
        "ram_gib": ram,
        "arch": platform.machine(),
        "system": platform.system(),
    }


def observed_threads() -> dict:
    import numba
    from threadpoolctl import threadpool_info

    return {
        "pools": sorted([i.get("internal_api"), i.get("num_threads")] for i in threadpool_info()),
        "numba": int(numba.config.NUMBA_NUM_THREADS),
        "env": {k: os.environ.get(k) for k in POOLS},
    }


def apply_threads(setting: str) -> None:
    """Must run before numpy, numba or any BLAS is imported."""
    for name in POOLS:
        if setting == "pinned":
            os.environ[name] = "1"
        else:
            os.environ.pop(name, None)


def threads_honoured(setting: str, seen: dict) -> str | None:
    if setting == "pinned":
        loose = [p for p in seen["pools"] if p[1] != 1]
        loose += [["numba", seen["numba"]]] if seen["numba"] != 1 else []
        unset = [k for k, v in seen["env"].items() if v != "1"]
        if loose or unset:
            return f"pinned requested but pools run wider: {loose}, unpinned variables {unset}"
        return None
    set_vars = {k: v for k, v in seen["env"].items() if v is not None}
    return f"default requested but pool variables are set: {set_vars}" if set_vars else None


# --------------------------------------------------------------------------- work counters

# (count name, module, function).  A target missing at a commit is reported with a null count.
COUNTERS = (
    ("gram_builds", "superglm.solvers.centered_system", "build_centered_system"),
    ("gram_table_passes", "superglm._group_matrix._lookup_tables", "table_centred_system"),
    ("leverage_table_passes", "superglm._group_matrix._lookup_tables", "table_leverages"),
    ("solved_row_passes", "superglm.reml._leverage_rows", "solved_row_blocks"),
    ("stop_passes", "superglm.reml.w_derivatives", "_leverage_derivative_terms"),
    ("tsqr_factorisations", "superglm.solvers.rank", "_tsqr_weighted_factor"),
    ("data_rank_factorisations", "superglm.solvers.rank", "decompose_factor"),
    ("gram_rank_decompositions", "superglm.solvers.rank", "_decompose_gram"),
    ("pirls_solves", "superglm.solvers.irls_direct", "fit_irls_direct"),
)
# Work counts read from superglm's own REML profile, under this record's names.
PROFILE_WORK = {
    "presolve_iterations": "reml_presolve_n_iter",
    "irls_calls": "irls_calls",
    "irls_iterations": "irls_iters",
    "line_search_refits": "reml_n_linesearch_fits",
    "line_search_full_evals": "reml_n_linesearch_full_evals",
    "w_corrections": "reml_n_w_correction",
}


class Counters:
    """Count outermost calls per kind on every thread; arguments and values pass through unchanged."""

    def __init__(self):
        self.calls: dict[str, int | None] = {}
        self._lock = threading.Lock()
        self._depth = threading.local()
        self._patched: list[tuple[object, str, object]] = []

    def _wrap(self, kind, fn):
        @functools.wraps(fn)
        def counted(*args, **kwargs):
            depth = getattr(self._depth, kind, 0)
            setattr(self._depth, kind, depth + 1)
            try:
                return fn(*args, **kwargs)
            finally:
                setattr(self._depth, kind, depth)
                if depth == 0:
                    with self._lock:
                        self.calls[kind] += 1

        return counted

    def install(self):
        import pkgutil

        import superglm

        for info in pkgutil.walk_packages(superglm.__path__, "superglm."):
            if "editor" not in info.name:
                try:
                    importlib.import_module(info.name)
                except Exception:  # noqa: BLE001, S110 - optional modules may not import here
                    pass
        for kind, module_name, attr in COUNTERS:
            module = sys.modules.get(module_name)
            original = getattr(module, attr, None) if module is not None else None
            if original is None:
                self.calls[kind] = None
                continue
            self.calls[kind] = 0
            counted = self._wrap(kind, original)
            for name, mod in list(sys.modules.items()):
                if (
                    name.startswith("superglm")
                    and mod is not None
                    and getattr(mod, attr, None) is original
                ):
                    setattr(mod, attr, counted)
                    self._patched.append((mod, attr, original))

    def uninstall(self):
        for mod, attr, original in reversed(self._patched):
            setattr(mod, attr, original)
        self._patched.clear()


# --------------------------------------------------------------------------- models and fits


def _glm_model(spec):
    import superglm
    from superglm import Categorical, Numeric, Spline, SuperGLM

    kinds = {"spline": Spline, "numeric": Numeric, "categorical": Categorical}
    features = {name: kinds[kind](**opts) for name, (kind, opts) in spec["features"].items()}
    family, family_opts = spec["family"]
    kwargs = dict(spec["model_options"])
    if spec["link"] is not None:
        kwargs["link"] = spec["link"]
    return SuperGLM(
        family=getattr(superglm.families, family)(**family_opts),
        features=features,
        interactions=[tuple(p) for p in spec["interactions"]],
        **kwargs,
    )


def _lss_model(spec):
    import superglm

    family = getattr(superglm, spec["family"][0])(**spec["family"][1])
    terms = {"s": superglm.s, "cat": superglm.cat, "re": superglm.re}
    predictors = [
        getattr(family, name)(*(terms[k](col, **o) for k, col, o in tl))
        for name, tl in spec["predictors"]
    ]
    return superglm.SuperLSS(family, *predictors, **spec["model_options"])


def _glm_inputs(spec, frame, rows=None):
    sl = slice(None) if rows is None else slice(0, rows)
    w, off = _column(frame, spec["weights"]), _column(frame, spec["offset"])
    return (
        frame.iloc[sl],
        _response(frame, spec["response"])[sl],
        None if w is None else w[sl],
        None if off is None else off[sl],
    )


def _lss_inputs(spec, frame, rows=None):
    sl = slice(None) if rows is None else slice(0, rows)
    w = _column(frame, spec["weights"])
    offsets = {k: _column(frame, v)[sl] for k, v in (spec["offsets"] or {}).items()} or None
    return (
        frame.iloc[sl],
        _response(frame, spec["response"])[sl],
        None if w is None else w[sl],
        offsets,
    )


def _warm_up(spec):
    """Compile and warm caches on the first training rows; never timed."""
    rows, iters = spec["warmup"]["rows"], spec["warmup"]["max_reml_iter"]
    if not rows:
        return None
    train, _ = load_book(spec["dataset"])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if spec["kind"] == "glm":
                X, y, w, off = _glm_inputs(spec, train, rows)
                _glm_model(spec).fit_reml(X, y, sample_weight=w, offset=off, max_reml_iter=iters)
            else:
                X, y, w, offsets = _lss_inputs(spec, train, rows)
                kw = {} if w is None else {"sample_weight": w}
                _lss_model(spec).fit_reml(X, y, offsets=offsets, max_reml_iter=iters, **kw)
    except Exception as exc:  # noqa: BLE001 - compilation is done even when a tiny warm-up fit refuses
        return f"{type(exc).__name__}: {exc}"[:300]
    return None


def _fit(spec, model, train) -> dict:
    if spec["kind"] == "glm":
        X, y, w, off = _glm_inputs(spec, train)
        method, opts = spec["fit"]
        result = getattr(model, method)(X, y, sample_weight=w, offset=off, **opts)
        return {"p_hat": float(result.p_hat)} if method == "estimate_p" else {}
    X, y, w, offsets = _lss_inputs(spec, train)
    model.fit_reml(X, y, offsets=offsets, **({} if w is None else {"sample_weight": w}))
    return {}


# --------------------------------------------------------------------------- held-out metrics


def _deciles(order_key, weights):
    import numpy as np

    order = np.argsort(order_key, kind="stable")
    cum = np.cumsum(weights[order]) / weights.sum()
    bins = np.empty(len(order), dtype=int)
    bins[order] = np.minimum((cum * 10).astype(int), 9)
    return bins


def _gini(actual, expected, weights):
    """Weighted Lorenz Gini of actual against predicted rate."""
    import numpy as np

    order = np.argsort(expected / weights, kind="stable")
    x = np.r_[0.0, np.cumsum(weights[order])] / weights.sum()
    lorenz = np.r_[0.0, np.cumsum(actual[order])] / actual.sum()
    return float(1.0 - np.sum(np.diff(x) * (lorenz[1:] + lorenz[:-1])))


def _ae(actual, expected, bins):
    return [
        float(actual[bins == b].sum() / expected[bins == b].sum())
        for b in range(int(bins.max()) + 1)
    ]


def glm_results(spec, model, test) -> dict:
    import numpy as np
    from scipy.special import gammaln

    from superglm.distributions import weighted_log_likelihood

    X, y, w, off = _glm_inputs(spec, test)
    w = np.ones(len(y)) if w is None else w
    mu = np.asarray(
        model.predict(X) if off is None else model.predict(X, offset=off), dtype=np.float64
    )
    dist = model._distribution
    out = {"test_deviance": float(np.sum(w * dist.deviance_unit(y, mu)))}
    if spec["nll"] == "poisson_counts":
        counts, lam = np.round(y * w), w * mu
        out["test_nll"] = float(np.sum(lam - counts * np.log(lam) + gammaln(counts + 1)))
    else:
        semantics = spec["model_options"].get("weight_semantics", "prior")
        try:
            phi = float(model.result.phi)
            out["test_nll"] = -float(
                weighted_log_likelihood(dist, y, mu, w, phi, weight_semantics=semantics)
            )
        except Exception as exc:  # noqa: BLE001 - a family without a density says why
            out["test_nll"] = None
            out["test_nll_unavailable"] = f"{type(exc).__name__}: {exc}"[:200]
    actual, expected = w * y, w * mu
    out["gini"] = _gini(actual, expected, w)
    ae = _ae(actual, expected, _deciles(mu, w))
    out["ae_deciles"] = ae
    out["ae_worst_dev"] = float(max(abs(v - 1.0) for v in ae))
    return out


def lss_results(spec, model, test) -> dict:
    import numpy as np
    from scipy import stats

    X, y, w, offsets = _lss_inputs(spec, test)
    kw = {} if w is None else {"sample_weight": w}
    scores = model.scores(X, y, which=("log",), offsets=offsets, **kw)
    out = {"test_nll": float(scores[scores.columns[0]].sum())}
    mu = np.asarray(model.predict(X, offsets=offsets), dtype=np.float64)
    params = model.predict_parameters(X, offsets=offsets)
    ww = np.ones(len(y)) if w is None else w
    family = spec["family"][0]
    pit = None
    if family == "NegativeBinomialLS":
        theta = params["theta"].to_numpy(np.float64)
        t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
        out["test_deviance"] = float(2.0 * np.sum(t - (y - mu)))
        expo = X["Exposure"].to_numpy(np.float64)
        bins = _deciles(mu / expo, expo)
        prob = theta / (theta + mu)
        lo, hi = stats.nbinom.cdf(y - 1, theta, prob), stats.nbinom.cdf(y, theta, prob)
        pit = lo + np.random.default_rng(SEED).uniform(size=len(y)) * (hi - lo)
    elif family == "TweedieLSS":
        pw = params["power"].to_numpy(np.float64)
        unit = (
            np.where(y > 0, y ** (2 - pw) / ((1 - pw) * (2 - pw)), 0.0)
            - y * mu ** (1 - pw) / (1 - pw)
            + mu ** (2 - pw) / (2 - pw)
        )
        out["test_deviance"] = float(np.sum(2.0 * ww * unit))
        bins = _deciles(mu, ww)
    else:
        out["test_deviance"] = float(2.0 * np.sum((y - mu) / mu - np.log(y / mu)))
        bins = _deciles(mu, ww)
        pit = np.asarray(model.predict_cdf(X, y, offsets=offsets), dtype=np.float64)
    ae = _ae(ww * y, ww * mu, bins)
    out["ae_deciles"] = ae
    out["ae_mean_dev"] = float(np.mean([abs(v - 1.0) for v in ae]))
    out["cov80"] = None if pit is None else float(np.mean((pit > 0.1) & (pit < 0.9)))
    if pit is None:
        out["cov80_unavailable"] = "no randomised PIT for the Tweedie point mass at zero"
    return out


def _scalars(d):
    return {
        k: v for k, v in (d or {}).items() if isinstance(v, (int, float, str, bool)) or v is None
    }


def _is_time(key):
    return key.endswith("_s") or key.endswith("_seconds") or "_s_" in key


def glm_state(model, calls) -> dict:
    profile = _scalars(getattr(model, "_reml_profile", None))
    diag = model.reml_diagnostics()
    work = dict(calls, outer_iterations=diag.get("n_reml_iter"))
    work.update({name: profile.get(key) for name, key in PROFILE_WORK.items()})
    dist = model._distribution
    return {
        "phases": {k: v for k, v in profile.items() if _is_time(k) and isinstance(v, (int, float))},
        "profile": {k: v for k, v in profile.items() if not _is_time(k)},
        "work": work,
        "backend": getattr(model.result, "direct_backend", None),
        "convergence": {
            "converged": diag.get("converged"),
            "termination_reason": diag.get("termination_reason"),
            **{k: v for k, v in profile.items() if "certif" in k or k.endswith("_converged")},
        },
        "fit": {
            "objective": diag.get("objective"),
            "edf": float(model.result.effective_df),
            "n_lambdas": len(diag.get("lambdas") or {}),
            "n_columns": int(model._dm.p),
            "theta": getattr(dist, "theta", None),
            "power": getattr(dist, "p", None),
            "phi": float(model.result.phi),
        },
    }


def lss_state(model, calls) -> dict:
    state = model._model.fit_state
    smoothing = state.smoothing
    profile = model.diagnose().profile
    snapshot = model._fit_phase_snapshot.as_dict() if model._fit_phase_snapshot is not None else {}
    work = dict(
        calls,
        outer_iterations=profile.outer_iterations,
        coefficient_fits=profile.coefficient_fits,
        inner_iterations=profile.inner_iterations,
        newton_iterations=int(smoothing.newton_iterations),
    )
    work.update({f"phase_count:{k}": v for k, v in (snapshot.get("counts") or {}).items()})
    edf = {k: float(v) for k, v in state.result.predictor_edf.items()}
    return {
        "phases": {k: float(v) for k, v in (snapshot.get("seconds") or {}).items()},
        "profile": {},
        "work": work,
        "backend": None,
        "convergence": {
            "converged": bool(smoothing.converged),
            "termination_reason": str(smoothing.convergence_reason),
        },
        "fit": {
            "objective": float(smoothing.objective),
            "edf": float(sum(edf.values())),
            "edf_by_predictor": edf,
        },
    }


# --------------------------------------------------------------------------- one run


def key_of(case, threads, sha, data_hash, env, mach) -> dict:
    return {
        "sha": sha,
        "dataset_hash": data_hash,
        "spec_hash": spec_hash(case),
        "environment": env,
        "machine": mach,
        "threads": threads,
    }


def case_key(case, threads) -> dict:
    import superglm

    sha = tree_state(Path(superglm.__file__).parent)["sha"]
    key = key_of(case, threads, sha, dataset_hash(CASES[case]["dataset"]), environment(), machine())
    return {"case": case, "key": key, "key_hash": digest(key)}


def _peak_rss_bytes() -> int:
    sys.path.insert(0, str(HERE))
    try:
        from _platform import peak_rss
    finally:
        sys.path.remove(str(HERE))
    return peak_rss().bytes


def run_case(case: str, threads: str, *, counts_only: bool) -> tuple[dict, dict]:
    """One complete fit of ``case``; returns its record and the fit's clock (for benchboard).  ``main`` applies the thread setting."""
    import numpy as np  # noqa: F401 - BLAS loads before the pools are read

    import superglm

    spec = CASES[case]
    seen = observed_threads()
    problem = threads_honoured(threads, seen)
    if problem and not counts_only:
        raise SystemExit(f"refused: {problem}")
    provenance = tree_state(Path(superglm.__file__).parent)
    provenance.update(superglm_file=superglm.__file__, superglm_version=superglm.__version__)
    harness = tree_state(HERE, paths=("benchmarks/books.py", "examples/fremtpl2_data.py"))
    train, test = load_book(spec["dataset"])
    data_hash = dataset_hash(spec["dataset"])
    env, mach = environment(), machine()

    warmup_error = _warm_up(spec)
    counters = Counters()
    counters.install()
    model = _glm_model(spec) if spec["kind"] == "glm" else _lss_model(spec)
    status, extra = "ok", {}
    rss_before = _peak_rss_bytes()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        started, t0, c0 = time.time(), time.perf_counter(), time.process_time()
        try:
            extra = _fit(spec, model, train)
        except Exception as exc:  # noqa: BLE001 - a refusal is a result of the commit, recorded as such
            status = f"raised {type(exc).__name__}: {exc}"[:500]
        finally:
            wall, cpu = time.perf_counter() - t0, time.process_time() - c0
            clock = {"started_at": started, "ended_at": time.time(), "wall_s": wall}
            counters.uninstall()
    rss_peak = _peak_rss_bytes()
    calls = dict(counters.calls)
    tally: dict[tuple[str, str], int] = {}
    for w in caught:
        k = (w.category.__name__, str(w.message)[:200])
        tally[k] = tally.get(k, 0) + 1

    record = {
        "schema": SCHEMA,
        "case": case,
        "spec": spec,
        "spec_hash": spec_hash(case),
        "dataset": {
            "spec": spec["dataset"],
            "hash": data_hash,
            "n_train": len(train),
            "n_test": len(test),
        },
        "provenance": provenance,
        "harness": harness,
        "environment": env,
        "machine": mach,
        "threads": threads,
        "threads_observed": seen,
        "threads_problem": problem,
        "counts_only": counts_only,
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": status,
        "warmup_error": warmup_error,
        "warnings": [
            {"category": c, "message": m, "count": n} for (c, m), n in sorted(tally.items())
        ],
    }
    if not counts_only:
        record["measurements"] = {
            "wall_s": wall,
            "cpu_s": cpu,
            "peak_rss_mib": rss_peak / 2**20,
            "peak_rss_before_fit_mib": rss_before / 2**20,
        }
    if status == "ok":
        state = glm_state(model, calls) if spec["kind"] == "glm" else lss_state(model, calls)
        state["fit"].update(extra)
        if counts_only:
            state.pop("phases")
        record.update(state)
        record["results"] = (glm_results if spec["kind"] == "glm" else lss_results)(
            spec, model, test
        )
    else:
        record["work"] = calls
    record["key"] = key_of(case, threads, provenance["sha"], data_hash, env, mach)
    record["key_hash"] = digest(record["key"])
    return record, clock


# --------------------------------------------------------------------------- records and benchboard


def _append_index(out: Path, entry: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "index.jsonl", "a", encoding="utf-8") as fh:
        fh.write(canonical(entry) + "\n")


def write_record(record: dict, out: Path) -> Path:
    """One JSON file per run plus a line in the append-only index."""
    sha = record["provenance"]["sha"] or "unknown"
    stamp = record["written"].replace(":", "").replace("-", "")
    suffix = "__counts" if record["counts_only"] else ""
    path = out / sha[:12] / f"{record['case']}__{record['threads']}__{stamp}{suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    entry = {
        k: record[k] for k in ("case", "threads", "counts_only", "status", "key_hash", "written")
    }
    entry.update(
        path=str(path), sha=record["provenance"]["sha"], dirty=record["provenance"]["dirty"]
    )
    _append_index(out, entry)
    return path


def bench_case(record_or_key: dict) -> str:
    """benchboard's case name: benchboard keys a baseline by (case, commit, hardware), so the
    case carries everything else in this harness's key -- dataset, spec, environment, machine
    and thread setting -- as one hash."""
    key = record_or_key.get("key", record_or_key)
    rest = {k: v for k, v in key.items() if k != "sha"}
    case = record_or_key.get("case") or "?"
    return f"books/{case}/{key['threads']}/{digest(rest)[:12]}"


def _finite(obj):
    """JSON-safe copy: benchboard stores strict JSON, so non-finite floats become null."""
    import math

    if isinstance(obj, dict):
        return {str(k): _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj if obj is None or isinstance(obj, (str, int, float, bool)) else str(obj)


def fit_payload(record: dict, clock: dict, task_id: str) -> dict:
    """benchboard's worker protocol v1: one fit; work counts and held-out metrics in the summary,
    the full record as the (compressed) diagnostics."""
    import uuid

    spec, fit = record["spec"], record.get("fit") or {}
    status = "success" if record["status"] == "ok" else "failed"
    phases = [
        {"name": k, "wall_s": float(v), "source": "superglm"}
        for k, v in (record.get("phases") or {}).items()
        if _finite(float(v)) is not None and v >= 0
    ]
    config = {
        "n_rows": record["dataset"]["n_train"],
        "n_features": len(spec.get("features") or {})
        or sum(len(t) for _, t in spec.get("predictors", [])),
        "distribution": spec["family"][0],
        "link": spec.get("link"),
        "model": "SuperGLM" if spec["kind"] == "glm" else "SuperLSS",
        "books_case": record["case"],
        "benchboard_case": bench_case(record),
        "threads": record["threads"],
        "sha": record["provenance"]["sha"],
        "dirty": record["provenance"]["dirty"],
        "counts_only": record["counts_only"],
        "key_hash": record["key_hash"],
        "spec_hash": record["spec_hash"],
        "dataset_hash": record["dataset"]["hash"],
    }
    summary = {
        "n_columns": fit.get("n_columns"),
        "solver": record.get("backend"),
        "converged": (record.get("convergence") or {}).get("converged"),
        "n_iter": (record.get("work") or {}).get("outer_iterations"),
        "objective": fit.get("objective"),
        "edf": fit.get("edf"),
        "work": record.get("work"),
        "results": {
            k: v for k, v in (record.get("results") or {}).items() if not isinstance(v, list)
        },
        "measurements": record.get("measurements"),
        "warnings": len(record.get("warnings") or []),
    }
    return {
        "protocol": 1,
        "task_id": task_id,
        "environment": dict(
            record["environment"], machine=record["machine"], threads=record["threads_observed"]
        ),
        "fits": [
            {
                "id": uuid.uuid4().hex,
                "parent_fit_id": None,
                "sequence": 0,
                "pid": os.getpid(),
                "ppid": os.getppid(),
                "method": f"books.{spec['fit'][0]}",
                "status": status,
                **clock,
                "config": _finite(config),
                "summary": _finite(summary),
                "phases": phases,
                "diagnostics": _finite(record),
            }
        ],
    }


def benchboard_dir() -> Path:
    return Path(os.environ.get("BENCHBOARD_DIR", Path.home() / ".cache" / "benchboard"))


def _benchboard_history():
    """benchboard's own history writer (benchboard_history.py, installed next to benchboard)."""
    import shutil

    exe = shutil.which("benchboard")
    for folder in filter(
        None, [os.environ.get("BENCHBOARD_HISTORY_DIR"), exe and str(Path(exe).resolve().parent)]
    ):
        if (Path(folder) / "benchboard_history.py").exists():
            if folder not in sys.path:
                sys.path.append(folder)
            return importlib.import_module("benchboard_history")
    return None


def record_to_benchboard(record: dict, clock: dict) -> str:
    """Write the fit into benchboard's fits, fit_phases and fit_diagnostics tables.

    Inside a benchboard job that captures fits (``BENCHBOARD_FIT_DIR`` set) the worker spool is
    published there and benchboard imports it when the job ends.  Otherwise this process records
    its own task, child of the enclosing benchboard task if any, through benchboard's writer.
    """
    import gzip
    import tempfile
    import uuid

    spool = os.environ.get("BENCHBOARD_FIT_DIR")
    task_id = os.environ.get("BENCHBOARD_TASK_ID")
    if spool and task_id:
        payload = gzip.compress(canonical(fit_payload(record, clock, task_id)).encode(), mtime=0)
        target = Path(spool) / f"books-{os.getpid()}-{uuid.uuid4().hex}.json.gz"
        tmp = target.with_suffix(".tmp")
        tmp.write_bytes(payload)
        tmp.replace(target)
        return f"spooled to {target}"
    history = _benchboard_history()
    if history is None:
        return "not recorded: benchboard_history.py not found next to benchboard"
    own = uuid.uuid4().hex
    db = history.History(benchboard_dir())
    db.start_task(
        {
            "id": own,
            "parent_task_id": task_id,
            "label": f"books-{record['case']}",
            "kind": "books",
            "description": f"books.py run {record['case']} --threads {record['threads']}",
            "launcher_pid": os.getppid(),
            "command_pid": os.getpid(),
            "cwd": os.getcwd(),
            "command": sys.argv,
            "started_at": clock["started_at"],
            "capture_enabled": True,
        }
    )
    with tempfile.TemporaryDirectory() as folder:
        payload = fit_payload(record, clock, own)
        (Path(folder) / "books.json.gz").write_bytes(
            gzip.compress(canonical(payload).encode(), mtime=0)
        )
        imported = db.import_spools(own, folder)
    ok = imported["fits"] == 1 and not imported["errors"]
    db.finish_task(
        own,
        status="success" if ok else "failed",
        ended_at=time.time(),
        wall_s=clock["wall_s"],
        capture_status="collected" if ok else "import_errors",
    )
    return (
        f"recorded as benchboard task {own}"
        if ok
        else f"benchboard import failed: {imported['errors']}"
    )


def stored_records(sha: str, case: str, threads: str) -> list[dict]:
    """Full records of this harness's fits at ``sha`` in benchboard's history, oldest first."""
    import sqlite3

    path = benchboard_dir() / "history.sqlite3"
    history = _benchboard_history()
    if history is None or not path.exists():
        return []
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    try:
        rows = db.execute(
            """SELECT d.codec, d.payload FROM fits f JOIN fit_diagnostics d ON d.fit_id = f.id
               WHERE f.method LIKE 'books.%' AND json_extract(f.config_json, '$.sha') = ?
               AND json_extract(f.config_json, '$.books_case') = ?
               AND json_extract(f.config_json, '$.threads') = ? ORDER BY f.started_at""",
            (sha, case, threads),
        ).fetchall()
    finally:
        db.close()
    return [history.decode_diagnostics(codec, blob) for codec, blob in rows]


def usable_baseline(rec: dict) -> bool:
    return rec["status"] == "ok" and not rec["counts_only"] and rec["provenance"]["dirty"] is False


# --------------------------------------------------------------------------- baseline


def _resolve(ref: str) -> str:
    sha = _git(HARNESS_ROOT, "rev-parse", "--verify", f"{ref}^{{commit}}")
    if not sha:
        raise SystemExit(f"cannot resolve {ref!r} in {HARNESS_ROOT}")
    return sha


def merge_base(sha: str = "HEAD") -> str:
    base = _git(HARNESS_ROOT, "merge-base", sha, UPSTREAM)
    if not base:
        raise SystemExit(
            f"no merge base between {sha} and {UPSTREAM}; name the baseline ref explicitly"
        )
    return base


def _venv_python(tree: Path) -> Path:
    return tree / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def prepare_tree(sha: str) -> Path:
    """A detached worktree at ``sha`` with its own synced environment; reused when present."""
    tree = TREES / sha[:12]
    if not (tree / ".git").exists():
        TREES.mkdir(parents=True, exist_ok=True)
        cmd = ["git", "-C", str(HARNESS_ROOT), "worktree", "add", "--detach", str(tree), sha]
        subprocess.run(cmd, check=True, timeout=600)
    if _git(tree, "rev-parse", "HEAD") != sha:
        raise SystemExit(f"{tree} is not at {sha}; remove it with `git worktree remove {tree}`")
    if not _venv_python(tree).exists():
        subprocess.run(
            ["uv", "sync", "--python", "3.13", "--extra", "dev"], cwd=tree, check=True, timeout=1800
        )
    return tree


def _clean_env() -> dict:
    return {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV")}


def _child(python: Path, tree: Path, args: list[str], timeout: float) -> int:
    cmd = [str(python), str(Path(__file__).resolve()), *args]
    try:
        return subprocess.run(
            cmd, cwd=tree, env=_clean_env(), timeout=timeout, check=False
        ).returncode
    except subprocess.TimeoutExpired:
        print(f"timed out after {timeout:.0f} s: {' '.join(args)}", file=sys.stderr)
        return 124


# Upper bounds for one benchboard run-timed lease, minutes (at least twice the expected time).
LEASE_MINUTES = {"std_mains": 2, "std_pairs4": 5, "std_pairs4_binned": 5, "std_pairs10": 12}


def timed_command(python: Path, case: str, threads: str, sha: str, args) -> list[str]:
    """The benchboard run-timed command for one baseline fit; pinned runs are --record'ed."""
    minutes = args.minutes or LEASE_MINUTES.get(case, 30)
    run = [
        str(python),
        str(Path(__file__).resolve()),
        "run",
        case,
        "--threads",
        threads,
        "--baseline",
    ]
    run = ["timeout", str(int(minutes * 90)), *run, "--out", str(args.out)]
    bench = ["benchboard", "run-timed", f"{args.label}-{case}-{threads}", str(minutes)]
    if threads == "pinned":
        # benchboard's --record refuses unless the command itself pins every pool.
        return [
            *bench,
            "--record",
            "{case}",
            "--commit",
            sha,
            "--",
            "env",
            *(f"{v}=1" for v in POOLS),
            *run,
        ]
    return [*bench, "--", *run]


def cmd_baseline(args) -> int:
    sha = _resolve(args.ref) if args.ref else merge_base()
    tree = prepare_tree(sha)
    if args.prepare:
        print(f"baseline tree ready: {tree} at {sha}")
        return 0
    python, cases, failures = _venv_python(tree), _case_list(args), 0
    for threads in args.threads.split(","):
        keys_file = Path(args.out) / "tmp" / f"keys_{sha[:12]}_{threads}.json"
        keys_file.parent.mkdir(parents=True, exist_ok=True)
        key_cmd = [
            str(python),
            str(Path(__file__).resolve()),
            "key",
            *cases,
            "--threads",
            threads,
            "--out",
            str(keys_file),
        ]
        key_cmd = [
            "benchboard",
            "heavy",
            f"{args.label}-keys",
            "--cores",
            "1",
            "--",
            "timeout",
            "1800",
            *key_cmd,
        ]
        if subprocess.run(key_cmd, cwd=tree, env=_clean_env(), check=False).returncode:
            raise SystemExit(f"key computation failed at {sha} ({threads})")
        for entry in json.loads(keys_file.read_text()):
            case, key = entry["case"], entry["key"]
            if key["sha"] != sha:
                raise SystemExit(f"the baseline tree imported superglm at {key['sha']}, not {sha}")
            stored = [
                r
                for r in stored_records(sha, case, threads)
                if usable_baseline(r) and r["key"] == key
            ]
            if stored and not args.force:
                print(f"reused  {case} [{threads}] {bench_case(entry)} at {sha[:12]}", flush=True)
                continue
            cmd = timed_command(python, case, threads, sha, args)
            cmd = [part.replace("{case}", bench_case(entry)) for part in cmd]
            rc = subprocess.run(cmd, cwd=tree, env=_clean_env(), check=False).returncode
            failures += rc != 0
            print(
                f"{'stored' if rc == 0 else f'FAILED ({rc})'}  {case} [{threads}] {bench_case(entry)}",
                flush=True,
            )
    return 1 if failures else 0


# --------------------------------------------------------------------------- compare


def _diff(a, b, prefix="") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += _diff(a.get(k), b.get(k), f"{prefix}.{k}" if prefix else k)
        return out
    return [] if a == b else [f"{prefix}: baseline {canonical(a)} vs branch {canonical(b)}"]


def match_baseline(record: dict, candidates: list[dict], base_sha: str):
    """(baseline, None) when a stored record's key equals the branch's in everything but the SHA;
    else (None, the reason, naming each component that differs)."""
    usable = [c for c in candidates if usable_baseline(c)]
    if not usable:
        return None, f"no stored baseline at {base_sha[:12]} for this case and thread setting"
    want = dict(record["key"], sha=base_sha)
    for c in reversed(usable):
        if c["key"] == want:
            return c, None
    strip = lambda k: {n: v for n, v in k.items() if n != "sha"}  # noqa: E731
    return None, "key differs from the stored baseline: " + "; ".join(
        _diff(strip(usable[-1]["key"]), strip(want))
    )


def _fmt(v, digits=1):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.{digits}f}" if abs(v) >= 1 or v == 0 else f"{v:.4g}"
    return str(v)


def _ratio(a, b):
    ok = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (a, b)) and a
    return f"{b / a:.2f}×" if ok else ""


def _arrow(a, b, digits=1, ratio=False):
    text = f"{_fmt(a, digits)} → {_fmt(b, digits)}"
    r = _ratio(a, b) if ratio else ""
    return f"{text} ({r})" if r else text


RESULT_COLUMNS = [
    ("objective", "REML objective", 4),
    ("edf", "EDF", 2),
    ("test_deviance", "Test deviance", 2),
    ("test_nll", "Test NLL", 2),
    ("gini", "Gini", 4),
    ("ae_worst_dev", "Worst-decile |A/E−1|", 4),
    ("ae_mean_dev", "Mean |A/E−1|", 4),
    ("cov80", "80% coverage", 4),
]


def _value(rec, field):
    return ((rec.get("fit") if field in ("objective", "edf") else rec.get("results")) or {}).get(
        field
    )


def _convergence(rec):
    c = rec.get("convergence") or {}
    return f"{c.get('converged')} ({c.get('termination_reason')})"


def comparison_table(pairs) -> str:
    cols = [c for c in RESULT_COLUMNS if any(_value(r, c[0]) is not None for p in pairs for r in p)]
    head = ["Case", "Threads", "Wall s", "CPU s", "Peak RSS MiB", "Outer it."] + [
        c[1] for c in cols
    ]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for base, branch in pairs:
        bm, nm = base.get("measurements") or {}, branch.get("measurements") or {}
        bw, nw = base.get("work") or {}, branch.get("work") or {}
        row = [
            branch["case"],
            branch["threads"],
            _arrow(bm.get("wall_s"), nm.get("wall_s"), 1, True),
            _arrow(bm.get("cpu_s"), nm.get("cpu_s"), 1, True),
            _arrow(bm.get("peak_rss_mib"), nm.get("peak_rss_mib"), 0, True),
            _arrow(bw.get("outer_iterations"), nw.get("outer_iterations")),
        ]
        row += [_arrow(_value(base, f), _value(branch, f), d) for f, _, d in cols]
        lines.append("| " + " | ".join(row) + " |")
    for base, branch in pairs:
        lines += [
            "",
            f"**{branch['case']} [{branch['threads']}]**: phases (≥ 0.05 s) and work counts",
            "",
        ]
        lines += ["| Item | Base | Branch | Ratio |", "|---|---|---|---|"]
        bp, np_ = base.get("phases") or {}, branch.get("phases") or {}
        for k in sorted(set(bp) | set(np_)):
            a, b = bp.get(k), np_.get(k)
            if max(abs(a or 0), abs(b or 0)) >= 0.05:
                lines.append(f"| {k} (s) | {_fmt(a, 2)} | {_fmt(b, 2)} | {_ratio(a, b)} |")
        bw, nw = base.get("work") or {}, branch.get("work") or {}
        for k in sorted(set(bw) | set(nw)):
            lines.append(
                f"| {k} | {_fmt(bw.get(k))} | {_fmt(nw.get(k))} | {_ratio(bw.get(k), nw.get(k))} |"
            )
        lines.append(
            f"| backend | {base.get('backend') or '—'} | {branch.get('backend') or '—'} | |"
        )
        lines.append(f"| converged | {_convergence(base)} | {_convergence(branch)} | |")
        lines.append(f"| status | {base['status']} | {branch['status']} | |")
        lines.append(
            f"| warnings | {len(base.get('warnings') or [])} | {len(branch.get('warnings') or [])} | |"
        )
    return "\n".join(lines)


def compare(records: list[dict], base_sha: str | None = None, lookup=stored_records):
    """(markdown table, []) when every record has a matching stored baseline; else (None, reasons)."""
    pairs, refusals = [], []
    for rec in records:
        base = base_sha or merge_base(rec["provenance"]["sha"])
        match, reason = match_baseline(rec, lookup(base, rec["case"], rec["threads"]), base)
        if reason:
            refusals.append(f"{rec['case']} [{rec['threads']}]: {reason}")
        else:
            pairs.append((match, rec))
    return (None, refusals) if refusals else (comparison_table(pairs), [])


def _latest_records(out: Path, sha: str, threads: str | None) -> list[Path]:
    latest: dict[tuple[str, str], dict] = {}
    index = out / "index.jsonl"
    if not index.exists():
        return []
    for line in index.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry["sha"] == sha and threads in (None, entry["threads"]):
            latest[(entry["case"], entry["threads"])] = entry
    return [Path(e["path"]) for e in latest.values()]


def cmd_compare(args) -> int:
    out = Path(args.out)
    if args.records:
        paths = [Path(p) for p in args.records]
    else:
        head = tree_state(HARNESS_ROOT)["sha"]
        paths = _latest_records(out, head, args.threads)
        if not paths:
            raise SystemExit(f"no records for {head[:12]} in {out}; run `books.py run CASE` first")
    records = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    table, refusals = compare(records, _resolve(args.base) if args.base else None)
    if refusals:
        print("refused:", file=sys.stderr)
        for r in refusals:
            print(f"  {r}", file=sys.stderr)
        return 2
    dirty = [r["case"] for r in records if r["provenance"]["dirty"]]
    if dirty:
        print(f"note: branch records from a tree with uncommitted changes: {', '.join(dirty)}\n")
    print(table)
    return 0


# --------------------------------------------------------------------------- CLI


def _case_list(args) -> list[str]:
    names = []
    for item in (args.set or "").split(",") + (args.cases or "").split(","):
        if item:
            names += SETS.get(item, [item])
    unknown = [n for n in names if n not in CASES]
    if unknown or not names:
        raise SystemExit(f"unknown or empty case list: {unknown or names}; see `books.py cases`")
    return list(dict.fromkeys(names))


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cases", help="list the case sets")

    p = sub.add_parser("run", help="one fit in this process")
    p.add_argument("case", choices=sorted(CASES))
    p.add_argument("--threads", choices=THREADS, default="pinned")
    p.add_argument("--counts-only", action="store_true", help="development: no timing fields")
    p.add_argument(
        "--baseline", action="store_true", help="refuse unless timed on a clean, committed tree"
    )
    p.add_argument("--no-benchboard", action="store_true", help="write the JSON record only")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--tree", help="run with another worktree's .venv and superglm")

    p = sub.add_parser("key", help="print the keys of cases without fitting")
    p.add_argument("cases", nargs="+")
    p.add_argument("--threads", choices=THREADS, default="pinned")
    p.add_argument("--out")

    p = sub.add_parser(
        "baseline", help="time a case set at REF through benchboard, reusing stored keys"
    )
    p.add_argument("ref", nargs="?", help=f"default: the merge base of HEAD with {UPSTREAM}")
    p.add_argument("--set", default="standard")
    p.add_argument("--cases")
    p.add_argument("--threads", default="pinned,default")
    p.add_argument("--force", action="store_true", help="re-time keys already stored")
    p.add_argument(
        "--prepare", action="store_true", help="only create and sync the baseline worktree"
    )
    p.add_argument("--label", default="books", help="benchboard label prefix (your agent name)")
    p.add_argument(
        "--minutes", type=float, help="benchboard lease bound per fit (default per case)"
    )
    p.add_argument("--out", default=str(DEFAULT_OUT))

    p = sub.add_parser("compare", help="branch records against the stored baseline")
    p.add_argument("records", nargs="*", help="record files; default: the latest per case for HEAD")
    p.add_argument(
        "--base", help=f"baseline ref; default: each record's merge base with {UPSTREAM}"
    )
    p.add_argument("--threads", choices=THREADS)
    p.add_argument("--out", default=str(DEFAULT_OUT))

    args = parser.parse_args(argv)
    if args.cmd == "cases":
        for name, members in SETS.items():
            print(f"{name}: {', '.join(members)}")
        return 0
    if args.cmd == "baseline":
        return cmd_baseline(args)
    if args.cmd == "compare":
        return cmd_compare(args)
    if args.cmd == "run" and args.tree:
        tree = Path(args.tree).resolve()
        i = argv.index("--tree")
        return _child(_venv_python(tree), tree, argv[:i] + argv[i + 2 :], 24 * 3600)
    apply_threads(args.threads)
    if args.cmd == "key":
        text = json.dumps([case_key(c, args.threads) for c in args.cases], indent=1, default=str)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        else:
            print(text)
        return 0
    record, clock = run_case(args.case, args.threads, counts_only=args.counts_only)
    if args.baseline and not usable_baseline(record):
        raise SystemExit(
            "refused: a baseline is a timed, successful fit of a clean, committed tree"
        )
    path = write_record(record, Path(args.out))
    where = "not recorded in benchboard (--no-benchboard)"
    if not args.no_benchboard:
        try:
            where = record_to_benchboard(record, clock)
        except Exception as exc:  # noqa: BLE001 - recording never replaces the fit's own outcome
            where = f"benchboard recording failed: {type(exc).__name__}: {exc}"
    summary = {
        k: record.get(k) for k in ("case", "threads", "status", "measurements", "work", "fit")
    }
    print(
        json.dumps(
            dict(summary, path=str(path), benchboard=where, benchboard_case=bench_case(record)),
            default=str,
        )
    )
    return 0 if record["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
