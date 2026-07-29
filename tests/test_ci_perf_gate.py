"""Tests for the CI performance gate (benchmarks/ci_perf_gate.py).

The gate protects the landed perf wins (audit 2026-07-28 §J.4 item 3 /
RFC-15): it compares measured benchmark medians and ratios against the
committed reference-box baselines, with generous headroom so slower CI
runners do not flap. A missing or failed benchmark case must fail the
gate — silence is not success.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_GATE_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "ci_perf_gate.py"


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("ci_perf_gate", _GATE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


@pytest.fixture()
def baselines():
    return {
        "tensor_cost": {
            "reference_seconds": {
                "tensor_cost_base_exact": 0.5,
                "tensor_cost_base_discrete": 0.2,
                "tensor_cost_ti_exact": 1.5,
                "tensor_cost_ti_discrete": 0.5,
            },
            "checks": {
                "tensor_multiplier_exact": {"max": 5.0},
                "tensor_multiplier_discrete": {"max": 5.0},
                "absolute_multiple": {"max": 8.0},
            },
        },
        "flagship": {
            "reference_median_s": 0.85,
            "checks": {"absolute_multiple": {"max": 8.0}},
        },
    }


def _tensor_measurement(base_exact=0.5, ti_exact=1.5, base_discrete=0.2, ti_discrete=0.5):
    cases = [
        {"tag": "tensor_cost_base_exact", "seconds": base_exact, "ok": True},
        {"tag": "tensor_cost_base_discrete", "seconds": base_discrete, "ok": True},
        {"tag": "tensor_cost_ti_exact", "seconds": ti_exact, "ok": True},
        {"tag": "tensor_cost_ti_discrete", "seconds": ti_discrete, "ok": True},
    ]
    summary = {
        "tensor_multiplier_exact": ti_exact / base_exact,
        "tensor_multiplier_discrete": ti_discrete / base_discrete,
    }
    return {"cases": cases, "summary": summary}


class TestEvaluateGate:
    def test_all_within_thresholds_passes(self, gate, baselines):
        checks = gate.evaluate_gate(
            baselines, tensor=_tensor_measurement(), flagship={"median_s": 0.9}
        )
        assert checks
        assert all(c.passed for c in checks)

    def test_ratio_breach_fails(self, gate, baselines):
        # Exact tensor multiplier 3.0x baseline ratio: 15x vs limit 5x.
        checks = gate.evaluate_gate(
            baselines,
            tensor=_tensor_measurement(ti_exact=7.5),
            flagship={"median_s": 0.9},
        )
        failed = [c for c in checks if not c.passed]
        assert any("tensor_multiplier_exact" in c.name for c in failed)

    def test_absolute_multiple_breach_fails(self, gate, baselines):
        # Every case 10x its reference: ratios stay clean, multiples breach.
        checks = gate.evaluate_gate(
            baselines,
            tensor=_tensor_measurement(
                base_exact=5.0, ti_exact=15.0, base_discrete=2.0, ti_discrete=5.0
            ),
            flagship={"median_s": 0.9},
        )
        failed = [c for c in checks if not c.passed]
        assert any("tensor_cost_base_exact" in c.name for c in failed)

    def test_flagship_breach_fails(self, gate, baselines):
        checks = gate.evaluate_gate(
            baselines, tensor=_tensor_measurement(), flagship={"median_s": 10.0}
        )
        failed = [c for c in checks if not c.passed]
        assert any("flagship" in c.name for c in failed)

    def test_missing_case_fails(self, gate, baselines):
        measurement = _tensor_measurement()
        measurement["cases"] = [
            c for c in measurement["cases"] if c["tag"] != "tensor_cost_ti_exact"
        ]
        checks = gate.evaluate_gate(baselines, tensor=measurement, flagship={"median_s": 0.9})
        assert any(not c.passed for c in checks)

    def test_failed_case_fails(self, gate, baselines):
        measurement = _tensor_measurement()
        for case in measurement["cases"]:
            if case["tag"] == "tensor_cost_ti_discrete":
                case["ok"] = False
                case["seconds"] = None
        checks = gate.evaluate_gate(baselines, tensor=measurement, flagship={"median_s": 0.9})
        assert any(not c.passed for c in checks)

    def test_missing_summary_ratio_fails(self, gate, baselines):
        measurement = _tensor_measurement()
        measurement["summary"].pop("tensor_multiplier_exact")
        checks = gate.evaluate_gate(baselines, tensor=measurement, flagship={"median_s": 0.9})
        failed = [c for c in checks if not c.passed]
        assert any("tensor_multiplier_exact" in c.name for c in failed)


class TestCommittedBaselines:
    """The gate reads this file in CI; a missing or malformed file must fail
    here, at test time, not as a FileNotFoundError inside the perf-gate job."""

    def test_committed_baselines_file_is_valid(self):
        import json

        path = _GATE_PATH.parent / "results" / "ci_perf_baselines.json"
        assert path.exists(), "benchmarks/results/ci_perf_baselines.json must be committed"
        baselines = json.loads(path.read_text())

        tensor = baselines["tensor_cost"]
        for tag in (
            "tensor_cost_base_exact",
            "tensor_cost_base_discrete",
            "tensor_cost_ti_exact",
            "tensor_cost_ti_discrete",
        ):
            assert tensor["reference_seconds"][tag] > 0
        assert tensor["checks"]["tensor_multiplier_exact"]["max"] > 0
        assert tensor["checks"]["tensor_multiplier_discrete"]["max"] > 0
        assert tensor["checks"]["absolute_multiple"]["max"] > 0
        # The output invariants are optional to evaluate_gate, so their
        # presence in the committed file must be pinned here or the gate
        # can silently degrade to timing-only.
        assert tensor["n"] == 30000
        for tag in (
            "tensor_cost_base_exact",
            "tensor_cost_base_discrete",
            "tensor_cost_ti_exact",
            "tensor_cost_ti_discrete",
        ):
            assert tensor["expected_p"][tag] > 0
            assert tensor["expected_outputs"][tag]["deviance"] > 0
            assert tensor["expected_outputs"][tag]["effective_df"] > 0
        assert tensor["output_rtol"] > 0
        assert tensor["expected_backend"]
        flagship = baselines["flagship"]
        assert flagship["n"] == 678013
        assert flagship["reference_median_s"] > 0
        assert flagship["checks"]["absolute_multiple"]["max"] > 0
        assert flagship["expected"]["deviance"] > 0
        assert flagship["expected"]["effective_df"] > 0
        assert flagship["expected"]["rtol"] > 0
        assert flagship["expected"]["backend"]


class TestOutputInvariants:
    """Timing alone cannot certify a fit: a regression that gets faster by
    silently doing less (fewer coefficients, different fit, wrong path) must
    fail on the recorded numerical outputs."""

    @pytest.fixture()
    def baselines_with_outputs(self, baselines):
        baselines = dict(baselines)
        baselines["tensor_cost"] = dict(baselines["tensor_cost"])
        baselines["tensor_cost"]["n"] = 30000
        baselines["tensor_cost"]["expected_p"] = {
            "tensor_cost_base_exact": 41,
            "tensor_cost_base_discrete": 41,
            "tensor_cost_ti_exact": 122,
            "tensor_cost_ti_discrete": 122,
        }
        baselines["flagship"] = dict(baselines["flagship"])
        baselines["flagship"]["expected"] = {
            "deviance": 212055.4,
            "effective_df": 43.33,
            "rtol": 1e-3,
        }
        return baselines

    def _measurement_with_outputs(self):
        measurement = _tensor_measurement()
        for case in measurement["cases"]:
            case["n"] = 30000
            case["p"] = 122 if "ti" in case["tag"] else 41
        return measurement

    def _flagship(self, deviance=212055.4, effective_df=43.33):
        return {"median_s": 0.9, "deviance": deviance, "effective_df": effective_df}

    def test_matching_outputs_pass(self, gate, baselines_with_outputs):
        checks = gate.evaluate_gate(
            baselines_with_outputs,
            tensor=self._measurement_with_outputs(),
            flagship=self._flagship(),
        )
        assert all(c.passed for c in checks)

    def test_deviance_drift_fails(self, gate, baselines_with_outputs):
        checks = gate.evaluate_gate(
            baselines_with_outputs,
            tensor=self._measurement_with_outputs(),
            flagship=self._flagship(deviance=213000.0),
        )
        assert any("deviance" in c.name and not c.passed for c in checks)

    def test_effective_df_drift_fails(self, gate, baselines_with_outputs):
        checks = gate.evaluate_gate(
            baselines_with_outputs,
            tensor=self._measurement_with_outputs(),
            flagship=self._flagship(effective_df=39.0),
        )
        assert any("effective_df" in c.name and not c.passed for c in checks)

    def test_wrong_design_width_fails(self, gate, baselines_with_outputs):
        measurement = self._measurement_with_outputs()
        for case in measurement["cases"]:
            if case["tag"] == "tensor_cost_ti_exact":
                case["p"] = 41
        checks = gate.evaluate_gate(
            baselines_with_outputs, tensor=measurement, flagship=self._flagship()
        )
        assert any("ti_exact.p" in c.name and not c.passed for c in checks)

    def test_wrong_row_count_fails(self, gate, baselines_with_outputs):
        measurement = self._measurement_with_outputs()
        for case in measurement["cases"]:
            if case["tag"] == "tensor_cost_base_exact":
                case["n"] = 5000
        checks = gate.evaluate_gate(
            baselines_with_outputs, tensor=measurement, flagship=self._flagship()
        )
        assert any("base_exact.n" in c.name and not c.passed for c in checks)

    def test_output_checks_optional_when_unconfigured(self, gate, baselines):
        checks = gate.evaluate_gate(
            baselines, tensor=_tensor_measurement(), flagship={"median_s": 0.9}
        )
        assert all(c.passed for c in checks)


class TestFitOutputAndBackendInvariants:
    @pytest.fixture()
    def baselines_full(self, baselines):
        baselines = dict(baselines)
        tc = dict(baselines["tensor_cost"])
        tc["output_rtol"] = 5e-3
        tc["expected_outputs"] = {
            tag: {"deviance": 9800.0, "effective_df": 22.0}
            for tag in (
                "tensor_cost_base_exact",
                "tensor_cost_base_discrete",
                "tensor_cost_ti_exact",
                "tensor_cost_ti_discrete",
            )
        }
        tc["expected_backend"] = "gram"
        baselines["tensor_cost"] = tc
        baselines["flagship"] = dict(baselines["flagship"])
        baselines["flagship"]["expected"] = {
            "deviance": 212055.4,
            "effective_df": 43.33,
            "rtol": 1e-3,
            "backend": "gram",
        }
        return baselines

    def _measurement(self, deviance=9800.0, backend="gram"):
        measurement = _tensor_measurement()
        for case in measurement["cases"]:
            case["deviance"] = deviance
            case["effective_df"] = 22.0
            case["direct_backend"] = backend
        return measurement

    def _flagship(self, backend="gram"):
        return {
            "median_s": 0.9,
            "deviance": 212055.4,
            "effective_df": 43.33,
            "direct_backend": backend,
        }

    def test_matching_outputs_and_backend_pass(self, gate, baselines_full):
        checks = gate.evaluate_gate(
            baselines_full, tensor=self._measurement(), flagship=self._flagship()
        )
        assert all(c.passed for c in checks)

    def test_tensor_deviance_drift_fails(self, gate, baselines_full):
        checks = gate.evaluate_gate(
            baselines_full, tensor=self._measurement(deviance=9900.0), flagship=self._flagship()
        )
        assert any(
            "deviance" in c.name and "tensor_cost" in c.name and not c.passed for c in checks
        )

    def test_tensor_backend_mismatch_fails(self, gate, baselines_full):
        checks = gate.evaluate_gate(
            baselines_full, tensor=self._measurement(backend="qr"), flagship=self._flagship()
        )
        assert any("backend" in c.name and "tensor_cost" in c.name and not c.passed for c in checks)

    def test_flagship_backend_mismatch_fails(self, gate, baselines_full):
        checks = gate.evaluate_gate(
            baselines_full,
            tensor=self._measurement(),
            flagship=self._flagship(backend="structured"),
        )
        assert any(c.name == "flagship.backend" and not c.passed for c in checks)


class TestFlagshipRowCount:
    def test_row_count_drop_fails(self, gate, baselines):
        baselines = dict(baselines)
        baselines["flagship"] = dict(baselines["flagship"])
        baselines["flagship"]["n"] = 678013
        checks = gate.evaluate_gate(
            baselines,
            tensor=_tensor_measurement(),
            flagship={"median_s": 0.9, "n": 600000},
        )
        assert any(c.name == "flagship.n" and not c.passed for c in checks)

    def test_matching_row_count_passes(self, gate, baselines):
        baselines = dict(baselines)
        baselines["flagship"] = dict(baselines["flagship"])
        baselines["flagship"]["n"] = 678013
        checks = gate.evaluate_gate(
            baselines,
            tensor=_tensor_measurement(),
            flagship={"median_s": 0.9, "n": 678013},
        )
        assert all(c.passed for c in checks)
