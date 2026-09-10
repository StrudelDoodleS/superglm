"""Complete-fit comparison for the rank-deficient decomposition change.

AGENTS.md requires performance-sensitive work to compare complete-fit timing,
memory, numerical outputs and actual backend dispatch against the relevant
baseline.  The alias-representative change is performance-sensitive -- it is the
reason a 41-level `cat_cat` refit went from minutes to seconds -- so it needs
all four in one place rather than an eigendecomposition count and an isolated
timing.

Run it on each side of the change and diff the JSON::

    uv run python benchmarks/rank_deficient_complete_fit.py --repeats 1 --out branch.json
    # check out the baseline's src/, then
    uv run python benchmarks/rank_deficient_complete_fit.py --repeats 1 --out baseline.json

The fit is the deficient one the change exists for: a 41-level `cat_cat` pair
on 6,000 training rows, 1,680 parameters, 54 of them unidentifiable.

Three measurement caveats the payload carries explicitly rather than leaving to
a reader, because each was got wrong once:

* wall clock is min-of-N, since a shared machine's median is not reproducible;
* `ru_maxrss` is a high-water mark for the WHOLE process, so it compares across
  runs only when both took the same number of fits -- `peak_rss_measures_fits`
  records that, and its unit differs by platform, so `ru_maxrss_unit` records
  that too;
* Native pools are observed synchronously at solver events in a separate
  untimed run. Timed runs have no observer and retain first-use work in the fit.

`tests/test_rank_deficient_complete_fit.py` asserts those invariants of the
payload, which is what nothing was doing while five defects went by.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager, nullcontext
from pathlib import Path
from threading import get_ident

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info

import superglm
from superglm import SuperGLM
from superglm.features import Categorical

# `ru_maxrss` is bytes on macOS and kibibytes everywhere else.  Dividing by
# 1024 unconditionally is right on Linux and 1024x wrong on macOS, while still
# labelled MiB.
_RSS_UNIT = "bytes" if sys.platform == "darwin" else "kib"
_RSS_DIVISOR = 1024.0**2 if _RSS_UNIT == "bytes" else 1024.0


def _design(levels: int, rows: int, seed: int):
    rng = np.random.default_rng(seed)
    left = rng.integers(0, levels, rows)
    right = rng.integers(0, levels, rows)
    y = (
        rng.normal(scale=0.5, size=levels)[left]
        + rng.normal(scale=0.5, size=levels)[right]
        + rng.normal(scale=1.0, size=rows)
    )
    frame = pd.DataFrame({"g": [f"G{i}" for i in left], "h": [f"H{i}" for i in right]})
    train = np.random.default_rng(seed).permutation(rows)[: rows // 2]
    return frame.iloc[train], y[train]


def _pool_snapshot() -> list[dict[str, object]]:
    return sorted(
        (
            {
                "user_api": pool.get("user_api"),
                "internal_api": pool.get("internal_api"),
                "prefix": pool.get("prefix"),
                "version": pool.get("version"),
                "threading_layer": pool.get("threading_layer"),
                "num_threads": pool.get("num_threads"),
            }
            for pool in threadpool_info()
        ),
        key=lambda pool: (str(pool["prefix"]), str(pool["version"])),
    )


class _DispatchSampler:
    """Bounded native-pool snapshots at explicit fitting-thread events.

    Event counts 1, 2, 4, ... for each label and fit request a snapshot, up to
    a total cap. Frequencies describe those selected events, never time dwell.
    No thread is started: concurrent library enumeration can deadlock against
    first-use native module loading on the fitting thread.
    """

    def __init__(self, max_samples=256, max_configurations=64, max_event_names=64):
        if min(max_samples, max_configurations, max_event_names) < 1:
            raise ValueError("Native-pool observation limits must be positive.")
        self.max_samples = max_samples
        self.max_configurations = max_configurations
        self.max_event_names = max_event_names
        self._thread_id = None
        self._counts: dict[tuple, int] = {}
        self._events = Counter()
        self._sampled_events = Counter()
        self._session_events = Counter()
        self.samples = 0
        self.sample_attempts = 0
        self.events_seen = 0
        self.skipped_at_sample_cap = 0
        self.dropped_event_notifications = 0
        self.dropped_configuration_observations = 0
        self.error_count = 0
        self.errors = []
        self.fit_entries = 0
        self.fit_entries_with_samples = 0

    def sample(self, event: str) -> None:
        if self._thread_id != get_ident():
            raise RuntimeError("Native pools must be sampled on the active fitting thread.")
        self.events_seen += 1
        if event not in self._events and len(self._events) >= self.max_event_names:
            self.dropped_event_notifications += 1
            return
        self._events[event] += 1
        self._session_events[event] += 1
        count = self._session_events[event]
        if count & (count - 1):
            return
        if self.sample_attempts >= self.max_samples:
            self.skipped_at_sample_cap += 1
            return
        self.sample_attempts += 1
        try:
            keys = {
                tuple(sorted(pool.items(), key=lambda item: item[0])) for pool in _pool_snapshot()
            }
        except Exception as error:
            self.error_count += 1
            if len(self.errors) < 8:
                self.errors.append(f"{type(error).__name__}: {error}")
            return
        self.samples += 1
        self._sampled_events[event] += 1
        for key in keys:
            if key not in self._counts and len(self._counts) >= self.max_configurations:
                self.dropped_configuration_observations += 1
            else:
                self._counts[key] = self._counts.get(key, 0) + 1

    def __enter__(self) -> _DispatchSampler:
        if self._thread_id is not None:
            raise RuntimeError("Native-pool observation is already active.")
        self._thread_id = get_ident()
        self._session_events.clear()
        self._entry_samples = self.samples
        self.fit_entries += 1
        return self

    def __exit__(self, *_exc: object) -> None:
        self.fit_entries_with_samples += self.samples > self._entry_samples
        self._thread_id = None

    def observed(self) -> list[dict[str, object]]:
        """Configuration frequencies among sampled events, not elapsed time."""
        rows = []
        for key, count in self._counts.items():
            row = dict(key)
            row["samples_seen_in"] = count
            row["fraction_of_samples"] = round(count / max(self.samples, 1), 4)
            rows.append(row)
        return sorted(rows, key=lambda pool: (str(pool["prefix"]), str(pool["version"])))


def _native_pool_receipt(sampler):
    if sampler is None:
        return {
            "native_pools_during_fit": [],
            "native_pool_samples": 0,
            "native_pool_observation": {"status": "not_observed_in_timed_fit"},
        }
    pools = sampler.observed()
    status = (
        "observer_error"
        if sampler.error_count
        else "no_solver_events_observed"
        if not sampler.events_seen or not sampler.samples
        else "no_native_pools_observed"
        if not pools
        else "observed_at_solver_events"
    )
    return {
        "native_pools_during_fit": pools,
        "native_pool_samples": sampler.samples,
        "native_pool_observation": {
            "status": status,
            "semantics": "synchronous selected-event frequencies, not elapsed-time dwell",
            "sampling_policy": "event counts 1,2,4,... per label and fit, capped snapshot attempts",
            "events_seen": sampler.events_seen,
            "event_counts": dict(sampler._events),
            "sampled_event_counts": dict(sampler._sampled_events),
            "max_samples": sampler.max_samples,
            "sample_attempts": sampler.sample_attempts,
            "eligible_events_skipped_at_sample_cap": sampler.skipped_at_sample_cap,
            "max_configurations": sampler.max_configurations,
            "dropped_configuration_observations": sampler.dropped_configuration_observations,
            "max_event_names": sampler.max_event_names,
            "dropped_event_notifications": sampler.dropped_event_notifications,
            "error_count": sampler.error_count,
            "errors": sampler.errors,
            "fit_entries": sampler.fit_entries,
            "fit_entries_with_samples": sampler.fit_entries_with_samples,
        },
    }


@contextmanager
def _native_pool_dispatch(sampler, targets):
    """Observe existing numerical call/return events without replacing functions."""
    codes = {function.__code__: name for name, function in targets.items()}

    def observe(frame, event, _result):
        if event in {"call", "return"} and frame.f_code in codes:
            sampler.sample(f"{codes[frame.f_code]}:{event}")

    previous = sys.getprofile()
    sys.setprofile(observe)
    try:
        yield
    finally:
        sys.setprofile(previous)


def measure(
    levels: int, rows: int, repeats: int, seed: int, *, measure_time=False
) -> dict[str, object]:
    """One complete-fit measurement, as the payload the artifact records."""
    frame, response = _design(levels, rows, seed)
    walls: list[float] = []
    model = None
    sampler = None if measure_time else _DispatchSampler()
    for _ in range(repeats):
        model = SuperGLM(
            family="gaussian",
            features={"g": Categorical(), "h": Categorical()},
            interactions=[("g", "h")],
        )
        if sampler is None:
            observer = nullcontext()
        else:
            from superglm.solvers import rank

            observer = _native_pool_dispatch(
                sampler,
                {"decompose_gram": rank.decompose_gram, "decompose_factor": rank.decompose_factor},
            )
        with sampler if sampler is not None else nullcontext(), observer:
            started = time.perf_counter() if measure_time else None
            model.fit(frame, response)
            if started is not None:
                walls.append(time.perf_counter() - started)

    if model is None:  # pragma: no cover - guarded at the flag
        raise SystemExit("no fit was run")
    result = model.result
    info = result.rank_info
    beta = np.asarray(result.beta, dtype=float)
    zeros = np.flatnonzero(beta == 0.0)
    pool_receipt = _native_pool_receipt(sampler)
    return {
        "wall_time_status": "measured" if measure_time else "unmeasured",
        "configuration": {
            "levels": levels,
            # What the design actually contains, which is <= `levels` whenever
            # the uniform draw missed one. Reporting only the request lets a
            # 26x26 design be labelled 41.
            "levels_realized": {
                "g": int(frame["g"].nunique()),
                "h": int(frame["h"].nunique()),
            },
            "rows": rows,
            "train_rows": int(len(frame)),
            "parameters": int(model._dm.p),
            "repeats": repeats,
            "seed": seed,
            "data_sha256": hashlib.sha256(
                frame.to_csv(index=False, float_format="%.17g").encode() + response.tobytes()
            ).hexdigest(),
        },
        "timing_seconds": {
            "min": round(min(walls), 4) if walls else None,
            "median": round(float(np.median(walls)), 4) if walls else None,
            "all": [round(wall, 4) for wall in walls],
        },
        "memory": {
            "peak_rss_measures_fits": repeats,
            "ru_maxrss_unit": _RSS_UNIT,
            "peak_rss_mib": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _RSS_DIVISOR, 1
            ),
        },
        "numerical_outputs": {
            "effective_df": round(float(result.effective_df), 9),
            "deviance": round(float(result.deviance), 9),
            "beta_l2": round(float(np.linalg.norm(beta)), 9),
            "n_zero": int(zeros.size),
            "zero_index_sum": int(zeros.sum()),
            "n_iter": int(result.n_iter),
        },
        "full_outputs": {
            "coefficients": beta.tolist(),
            "prediction": np.asarray(model.predict(frame)).tolist(),
            "effective_df": float(result.effective_df),
            "deviance": float(result.deviance),
            "converged": bool(result.converged),
        },
        # "backend dispatch" is two things: which decomposition route each
        # retained system took -- the branch this work replaced -- and which
        # BLAS actually serviced it.
        "backend_dispatch": {
            "data_method": info.data.method,
            "augmented_method": info.augmented.method,
            "coefficient_method": info.coefficient.method,
            "data_rank": int(info.data.rank),
            "augmented_rank": int(info.augmented.rank),
            "data_representative": info.data.pivots is not None,
            "augmented_representative": info.augmented.pivots is not None,
            "blas": {
                "numpy": np.__version__,
                "sampled_during_fit": pool_receipt["native_pool_samples"] > 0,
                "samples": pool_receipt["native_pool_samples"],
                "pools_during_fit": pool_receipt["native_pools_during_fit"],
                "observation": pool_receipt["native_pool_observation"],
            },
            "python": platform.python_version(),
        },
        # Provenance. Without this a payload cannot say which tree produced it:
        # two runs of the SAME source labelled baseline and branch satisfy every
        # invariant this file's tests assert, and the commits recorded beside
        # the committed artifact are typed in by hand.
        "provenance": _provenance(),
    }


def _provenance() -> dict[str, object]:
    """Identify the tree that produced this payload, from the tree itself.

    Read from the installed package and from git rather than accepted as a
    flag, so a mislabelled comparison is not merely discouraged but unavailable.
    """
    head, dirty = _git_state()
    package = Path(superglm.__file__).resolve().parent
    source_hashes = {
        str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*.py"))
    }
    return {
        "superglm_version": _superglm_version(),
        "superglm_path": str(Path(superglm.__file__).resolve().parent),
        "git_commit": head,
        "git_dirty": dirty,
        "source_digest": hashlib.sha256(
            json.dumps(source_hashes, sort_keys=True).encode()
        ).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _superglm_version() -> str | None:
    return getattr(superglm, "__version__", None)


def _git_state() -> tuple[str | None, bool | None]:
    """``(commit, dirty)`` for the tree the package was imported FROM."""
    package_root = Path(superglm.__file__).resolve().parent
    try:
        head = subprocess.run(
            ["git", "-C", str(package_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(package_root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None, None
    return head or None, bool(status)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, default=41)
    parser.add_argument("--rows", type=int, default=12_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=31337)
    parser.add_argument("--out", default=None)
    parser.add_argument("--measure-time", action="store_true")
    args = parser.parse_args()
    for name, value, minimum in (
        ("--levels", args.levels, 2),
        ("--rows", args.rows, 2),
        ("--repeats", args.repeats, 1),
        # A negative seed died eight NumPy frames deep in `default_rng`, naming
        # no flag. Every input this harness accepts is validated at the flag.
        ("--seed", args.seed, 0),
    ):
        if value < minimum:
            raise SystemExit(f"{name} must be >= {minimum}, got {value}")
    # `_design` keeps half the rows and draws levels UNIFORMLY, so realizing
    # every level is coupon-collector, not pigeonhole. The old bound was
    # `rows // 2 >= levels`, which at 41 levels blessed 82 rows -- where 41
    # draws realize about 26 distinct levels. The harness then reported
    # `levels: 41` for a design that is really 26x26, so it measured a smaller
    # problem than its own configuration block claims.
    #
    # E[draws to see all L] = L * H_L ~ L * (ln L + gamma); require twice that
    # so the shortfall is rare rather than merely expected-to-clear.
    harmonic = sum(1.0 / i for i in range(1, args.levels + 1))
    needed = 2 * int(math.ceil(args.levels * harmonic))
    if args.rows // 2 < needed:
        raise SystemExit(
            f"--rows {args.rows} keeps {args.rows // 2} training rows, which will not "
            f"realize all {args.levels} levels of each factor; needs at least {2 * needed} "
            f"(coupon-collector, ~L*ln L, not L). Otherwise the run reports "
            f"levels={args.levels} for a smaller realized design."
        )

    payload = measure(
        args.levels, args.rows, args.repeats, args.seed, measure_time=args.measure_time
    )
    text = json.dumps(payload, indent=1, sort_keys=True)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
