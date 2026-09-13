"""Run FAST default and Purify on the frozen PSST simulation grid.

Install interpret-core==0.7.8 with --no-deps in the benchmark environment.
Uses the original runner unchanged; its old/corrected slots mean FAST
default/Purify here. The companion metadata records this mapping explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import psst_detection_study as study
from interpret import __version__ as interpret_version
from interpret.utils import measure_interactions
from interpret.utils._native import InteractionDetector, Native


@contextmanager
def native_flags(purify):
    original = InteractionDetector.calc_interaction_strength
    observed = Counter()

    def measured(self, feature_idxs, calc_interaction_flags, *args, **kwargs):
        if calc_interaction_flags != Native.CalcInteractionFlags_Default:
            raise RuntimeError("Unexpected public FAST flags")
        flags = calc_interaction_flags | (Native.CalcInteractionFlags_Purify if purify else 0)
        observed[flags] += 1
        return original(self, feature_idxs, flags, *args, **kwargs)

    InteractionDetector.calc_interaction_strength = measured
    try:
        yield observed
    finally:
        InteractionDetector.calc_interaction_strength = original


def fast_screen(model, frame, y, verify_old=False):
    family = type(model._distribution).__name__
    if family not in ("Gaussian", "Poisson"):
        raise ValueError(f"Unsupported study family {family}")
    eta = model._predict_eta_exact(frame)
    mean = eta if family == "Gaussian" else np.exp(eta)
    np.testing.assert_allclose(mean, model.predict(frame), rtol=8 * np.finfo(float).eps, atol=0)
    objective = "rmse" if family == "Gaussian" else "poisson_deviance"
    pairs = list(itertools.combinations(range(frame.shape[1]), 2))
    kwargs = dict(
        interactions=pairs,
        init_score=eta,
        feature_names=list(frame.columns),
        feature_types=["continuous"] * 12 + ["nominal"] * 18,
        max_interaction_bins=64,
        min_samples_leaf=4,
        min_hessian=0.0001,
        reg_alpha=0.0,
        reg_lambda=0.0,
        max_delta_step=0.0,
        objective=objective,
    )
    ranked, dispatch = {}, {"objective": objective, "baseline_eta_sum": float(np.sum(eta))}
    for method, pure in (("old", False), ("corrected", True)):
        with native_flags(pure) as flags:
            scores = measure_interactions(frame, y, **kwargs)
        expected = (
            Native.CalcInteractionFlags_Purify if pure else Native.CalcInteractionFlags_Default
        )
        if dict(flags) != {expected: 435} or len(scores) != 435:
            raise RuntimeError(f"Expected 435 FAST calls at flags {expected}; got {dict(flags)}")
        rows = [
            {"pair": ":".join(frame.columns[i] for i in pair), "z": float(strength), "rung": None}
            for pair, strength in scores
        ]
        if len({r["pair"] for r in rows}) != 435 or not all(np.isfinite(r["z"]) for r in rows):
            raise RuntimeError("FAST candidate identity/finite-score check failed")
        rows.sort(key=lambda row: (-row["z"], row["pair"]))
        ranked[method] = rows
        dispatch["purify_calls" if pure else "default_calls"] = sum(flags.values())
    if verify_old:
        unwrapped = measure_interactions(frame, y, **kwargs)
        direct = {
            ":".join(frame.columns[i] for i in pair): float(value) for pair, value in unwrapped
        }
        if any(row["z"] != direct[row["pair"]] for row in ranked["old"]):
            raise RuntimeError("Instrumented FAST default differs from unwrapped public function")
    return ranked, dispatch


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "interpret_core": interpret_version,
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(
            Path(__file__).with_name("psst_fast_protocol.md").read_bytes()
        ).hexdigest(),
        "methods": {"old": "FAST default", "corrected": "FAST Purify"},
        "reference": "https://interpret.ml/docs/python/api/measure_interactions.html",
        "native_purify_flag": Native.CalcInteractionFlags_Purify,
        "baseline": "Same SuperGLM additive fit and link-scale predictions as PSST",
    }
    metadata_path = args.output / "fast-metadata.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        raise RuntimeError("FAST metadata differs; use a fresh output directory")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    study.screen = fast_screen
    study.main()


if __name__ == "__main__":
    main()
