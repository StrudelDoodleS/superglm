"""One complete timed run for the Tweedie/NB2 profiling rebuild's receipts.

Runs one ``estimate_p``, or one ``fit_reml`` at a fixed power, on a case from
``tweedie_nb_characterisation.build_case`` and prints one JSON line: wall and
CPU seconds, peak RSS, the estimate, the candidate-fit count and the backend
the published fit dispatched. It calls public API; time the pre-rebuild code
with that tree's own copy of this driver, whose bucket targets and ``--method``
option name the pre-rebuild functions. Run each repetition in a fresh process
(``ru_maxrss`` is a process high-water mark), with the thread pools pinned,
interleaving cases from outside:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    uv run python benchmarks/tweedie_estimate_p_timing.py --case positive96 --fit-mode reml

``--buckets`` is the separate profiling run. It wraps each target in
``BUCKET_TARGETS`` and charges wall time to candidate fits, the phi/density
profile or neither, a nested call counting toward the outermost bucket
entered. ``reml_scale`` is timed on its own clock: it runs inside candidate
fits. The output lists any target this tree lacks, so the list is updated
rather than silently measuring nothing.
"""

from __future__ import annotations

import argparse
import functools
import importlib
import json
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import superglm
from superglm import Tweedie

if __package__:
    from benchmarks import _platform
    from benchmarks.tweedie_nb_characterisation import build_case
else:  # run by filename
    import _platform
    from tweedie_nb_characterisation import build_case

# (module, attribute, bucket). ML candidates fit through _solve_coefficients and
# REML candidates through SuperGLM.fit_reml; the profile_ops publication refit
# goes through the fit_ops workspace entries and re-profiles phi there. The
# published fit statistics evaluate the density outside any fit, as the
# pre-rebuild targets' density evaluator did.
BUCKET_TARGETS = (
    ("superglm.profiling.tweedie", "_solve_coefficients", "candidate_fits"),
    ("superglm.model.api", "SuperGLM.fit_reml", "candidate_fits"),
    ("superglm.model.fit_ops", "_fit_reml_in_workspace", "candidate_fits"),
    ("superglm.model.fit_ops", "_fit_in_workspace", "candidate_fits"),
    ("superglm.profiling.tweedie", "profile_phi_at", "phi_density"),
    ("superglm.model.profile_ops", "profile_phi_at", "phi_density"),
    ("superglm._tweedie", "tweedie_logpdf_pair", "phi_density"),
    ("superglm._tweedie", "tweedie_logpdf", "phi_density"),
    ("superglm.reml.objective", "profile_tweedie_reml_scale", "reml_scale"),
    ("superglm.reml.direct", "profile_tweedie_reml_scale", "reml_scale"),
    ("superglm.reml.discrete", "profile_tweedie_reml_scale", "reml_scale"),
)
_SEPARATE_CLOCKS = frozenset({"reml_scale"})


class _OutermostClock:
    """Wall seconds per bucket, charging nested calls to the outermost bucket entered."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self._running = False

    def wrap(self, bucket: str, function):
        @functools.wraps(function)
        def timed(*args, **kwargs):
            if self._running:
                return function(*args, **kwargs)
            self._running = True
            self.calls[bucket] += 1
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                self.seconds[bucket] += time.perf_counter() - started
                self._running = False

        return timed


def _install_buckets(stack: ExitStack, clocks: dict[str, _OutermostClock]) -> list[str]:
    missing = []
    for module_name, attribute, bucket in BUCKET_TARGETS:
        owner = importlib.import_module(module_name)
        *path, name = attribute.split(".")
        owner = functools.reduce(getattr, path, owner)
        function = getattr(owner, name, None)
        if function is None:
            missing.append(f"{module_name}.{attribute}")
            continue
        clock = clocks[bucket if bucket in _SEPARATE_CLOCKS else "main"]
        stack.enter_context(patch.object(owner, name, clock.wrap(bucket, function)))
    return missing


def _peak_rss_mb() -> float:
    return _platform.peak_rss().bytes / 2**20


def _backend(model) -> dict:
    solver = model._solver_pirls_result()
    return {
        "last_fit_meta": dict(model._last_fit_meta or {}),
        "direct_backend": str(getattr(solver, "direct_backend", None)),
    }


def _operation(model, X, y, sample_weight, offset, args):
    if args.fit_reml_at is not None:
        model.family = Tweedie(p=args.fit_reml_at)
        model.fit_reml(X, y, sample_weight=sample_weight, offset=offset)
        return None
    options = {"fit_mode": args.fit_mode}
    if args.search_fit_mode is not None:
        options["search_fit_mode"] = args.search_fit_mode
    return model.estimate_p(X, y, sample_weight, offset, **options)


def _outcome(model, result) -> dict:
    if result is None:
        return {"phi_hat": float(model.result.phi), "n_reml_iter": model._reml_result.n_reml_iter}
    return {
        "p_hat": float(result.p_hat),
        "phi_hat": float(result.phi_hat),
        "nll": float(result.nll),
        "n_candidates": len(result.evaluations),
    }


def run(model, X, y, sample_weight=None, offset=None, *, args) -> dict:
    """Time (or, with ``args.buckets``, profile) one operation on one model."""
    superglm.warmup()
    clocks = {"main": _OutermostClock(), "reml_scale": _OutermostClock()}
    with ExitStack() as stack:
        missing = _install_buckets(stack, clocks) if args.buckets else []
        rss_before = _peak_rss_mb()
        wall, cpu = time.perf_counter(), time.process_time()
        result = _operation(model, X, y, sample_weight, offset, args)
        wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
    record = {
        "superglm": str(Path(superglm.__file__).resolve().parent),
        "wall_s": wall,
        "cpu_s": cpu,
        "peak_rss_mb": _peak_rss_mb(),
        "peak_rss_before_mb": rss_before,
        "load_average": _platform.load_average(),
        **_outcome(model, result),
        **_backend(model),
    }
    if args.buckets:
        seconds = clocks["main"].seconds
        record["buckets_s"] = {
            "candidate_fits": seconds["candidate_fits"],
            "phi_density": seconds["phi_density"],
            "other": wall - seconds["candidate_fits"] - seconds["phi_density"],
            "reml_scale_inside_fits": clocks["reml_scale"].seconds["reml_scale"],
        }
        record["bucket_calls"] = dict(clocks["main"].calls) | dict(clocks["reml_scale"].calls)
        record["targets_missing"] = missing
    return record


def parser() -> argparse.ArgumentParser:
    parse = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parse.add_argument("--case", default=None, help="a build_case name")
    parse.add_argument("--fit-mode", default="fit", choices=("fit", "reml"))
    parse.add_argument("--search-fit-mode", default=None, choices=("fit", "reml"))
    parse.add_argument("--fit-reml-at", type=float, default=None, help="time fit_reml at this p")
    parse.add_argument("--buckets", action="store_true", help="the profiling run")
    return parse


def main() -> None:
    args = parser().parse_args()
    model, X, y = build_case(args.case)
    print(json.dumps({"case": args.case, **vars(args), **run(model, X, y, args=args)}))


if __name__ == "__main__":
    main()
