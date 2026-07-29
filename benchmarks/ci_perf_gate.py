"""CI performance gate: compare measured benchmark medians against baselines.

Protects the landed perf wins (audit 2026-07-28 §J.4 item 3 / RFC-15).
Reads the committed reference-box baselines plus the measured JSON outputs
of ``benchmark_tensor_cost.py`` and ``timing_30rep_superglm.py``, and fails
(exit 1) when any check breaches its threshold.

Threshold philosophy (benchmark canon: single-thread BLAS both sides,
medians not single runs):

- **Ratio checks** (tensor-fit / base-fit multipliers) are machine-speed
  invariant, so they carry the tight-ish limits.
- **Absolute checks** are expressed as generous multiples of the reference
  box's medians — CI runners are slower and noisier, and the gate exists to
  catch order-of-magnitude regressions (a lost 22x), not 20% noise.
- A missing or failed benchmark case fails the gate: silence is not success.

Usage::

    uv run python benchmarks/ci_perf_gate.py \
        --baselines benchmarks/results/ci_perf_baselines.json \
        --tensor-json /tmp/tensor_cost_ci.json \
        --flagship-json /tmp/flagship_ci.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

TENSOR_CASE_TAGS = (
    "tensor_cost_base_exact",
    "tensor_cost_base_discrete",
    "tensor_cost_ti_exact",
    "tensor_cost_ti_discrete",
)


@dataclass(frozen=True)
class GateCheck:
    name: str
    measured: float | None
    limit: float
    passed: bool

    def render(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        measured = "missing" if self.measured is None else f"{self.measured:.3f}"
        return f"  [{status}] {self.name}: measured {measured}, limit {self.limit:.3f}"


def _check(name: str, measured: float | None, limit: float) -> GateCheck:
    passed = measured is not None and float(measured) <= float(limit)
    return GateCheck(name=name, measured=measured, limit=float(limit), passed=passed)


def evaluate_gate(baselines: dict, tensor: dict, flagship: dict) -> list[GateCheck]:
    """Evaluate every configured check; missing data fails the relevant check."""
    checks: list[GateCheck] = []

    tensor_base = baselines["tensor_cost"]
    tensor_checks = tensor_base["checks"]
    summary = tensor.get("summary", {})
    for ratio_name in ("tensor_multiplier_exact", "tensor_multiplier_discrete"):
        limit = tensor_checks[ratio_name]["max"]
        checks.append(_check(f"tensor_cost.{ratio_name}", summary.get(ratio_name), limit))

    multiple = tensor_checks["absolute_multiple"]["max"]
    reference_seconds = tensor_base["reference_seconds"]
    cases_by_tag = {case.get("tag"): case for case in tensor.get("cases", [])}
    expected_p = tensor_base.get("expected_p")
    expected_n = tensor_base.get("n")
    expected_outputs = tensor_base.get("expected_outputs")
    output_rtol = float(tensor_base.get("output_rtol", 0.0))
    expected_backend = tensor_base.get("expected_backend")
    for tag in TENSOR_CASE_TAGS:
        case = cases_by_tag.get(tag)
        ok = bool(case and case.get("ok"))
        measured = case.get("seconds") if ok else None
        limit = reference_seconds[tag] * multiple
        checks.append(_check(f"tensor_cost.{tag}.median_s", measured, limit))
        # Output invariants: timing alone cannot certify a fit — a run that
        # got faster by silently doing less must fail on what it computed.
        if expected_p is not None:
            deviation = abs(case.get("p", -1) - expected_p[tag]) if ok else None
            checks.append(_check(f"tensor_cost.{tag}.p", deviation, 0.0))
        if expected_n is not None:
            deviation = abs(case.get("n", -1) - expected_n) if ok else None
            checks.append(_check(f"tensor_cost.{tag}.n", deviation, 0.0))
        if expected_outputs is not None:
            for field in ("deviance", "effective_df"):
                measured_value = case.get(field) if ok else None
                deviation = (
                    abs(measured_value / expected_outputs[tag][field] - 1.0)
                    if measured_value is not None
                    else None
                )
                checks.append(_check(f"tensor_cost.{tag}.{field}.rel_dev", deviation, output_rtol))
        if expected_backend is not None:
            match = 0.0 if ok and case.get("direct_backend") == expected_backend else None
            checks.append(_check(f"tensor_cost.{tag}.backend", match, 0.0))

    flagship_base = baselines["flagship"]
    flagship_multiple = flagship_base["checks"]["absolute_multiple"]["max"]
    checks.append(
        _check(
            "flagship.median_s",
            flagship.get("median_s"),
            flagship_base["reference_median_s"] * flagship_multiple,
        )
    )
    flagship_expected = flagship_base.get("expected")
    if flagship_expected is not None:
        rtol = float(flagship_expected["rtol"])
        for field in ("deviance", "effective_df"):
            measured_value = flagship.get(field)
            deviation = (
                abs(measured_value / flagship_expected[field] - 1.0)
                if measured_value is not None
                else None
            )
            checks.append(_check(f"flagship.{field}.rel_dev", deviation, rtol))
        backend_expected = flagship_expected.get("backend")
        if backend_expected is not None:
            match = 0.0 if flagship.get("direct_backend") == backend_expected else None
            checks.append(_check("flagship.backend", match, 0.0))
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", required=True)
    parser.add_argument("--tensor-json", required=True)
    parser.add_argument("--flagship-json", required=True)
    args = parser.parse_args()

    baselines = json.loads(Path(args.baselines).read_text())
    tensor = json.loads(Path(args.tensor_json).read_text())
    flagship = json.loads(Path(args.flagship_json).read_text())

    checks = evaluate_gate(baselines, tensor=tensor, flagship=flagship)
    print("CI performance gate:")
    for check in checks:
        print(check.render())
    failed = [check for check in checks if not check.passed]
    if failed:
        print(f"\n{len(failed)} check(s) FAILED against {args.baselines}")
        return 1
    print(f"\nAll {len(checks)} checks passed against {args.baselines}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
