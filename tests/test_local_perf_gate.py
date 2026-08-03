"""Synthetic tests for the fail-closed, named-profile local performance gate."""

from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_GATE_PATH = _ROOT / "benchmarks" / "local_perf_gate.py"
_BASELINE_PATH = _ROOT / "benchmarks" / "results" / "local_perf_baselines.json"
_PROFILE_ID = "test-reference-profile"
_TENSOR_REPS = 3
_FLAGSHIP_REPS = 4


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("local_perf_gate", _GATE_PATH)
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
        "machine_profile_id": _PROFILE_ID,
        "certification_enabled": True,
        "tensor_cost": {
            "n": 30000,
            "reps": _TENSOR_REPS,
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
            "expected_p": {
                "tensor_cost_base_exact": 41,
                "tensor_cost_base_discrete": 41,
                "tensor_cost_ti_exact": 122,
                "tensor_cost_ti_discrete": 122,
            },
            "expected_outputs": {
                tag: {"deviance": 9800.0, "effective_df": 22.0}
                for tag in (
                    "tensor_cost_base_exact",
                    "tensor_cost_base_discrete",
                    "tensor_cost_ti_exact",
                    "tensor_cost_ti_discrete",
                )
            },
            "output_rtol": 5e-3,
            "expected_backend": "gram",
        },
        "flagship": {
            "n": 678013,
            "reps": _FLAGSHIP_REPS,
            "reference_median_s": 0.85,
            "checks": {"absolute_multiple": {"max": 8.0}},
            "expected": {
                "deviance": 212055.4,
                "effective_df": 43.33,
                "rtol": 1e-3,
                "backend": "gram",
            },
        },
    }


def _tensor_measurement(
    base_exact: float = 0.5,
    ti_exact: float = 1.5,
    base_discrete: float = 0.2,
    ti_discrete: float = 0.5,
) -> dict:
    case_values = (
        ("tensor_cost_base_exact", base_exact),
        ("tensor_cost_base_discrete", base_discrete),
        ("tensor_cost_ti_exact", ti_exact),
        ("tensor_cost_ti_discrete", ti_discrete),
    )
    cases = []
    for tag, seconds in case_values:
        with_tensor = "_ti_" in tag
        discrete = tag.endswith("_discrete")
        cases.append(
            {
                "tag": tag,
                "seconds": seconds,
                "ok": True,
                "tensor": with_tensor,
                "discrete": discrete,
                "reps": _TENSOR_REPS,
                "rep_seconds": [seconds - 0.001, seconds, seconds + 0.001],
                "warmup_seconds": [seconds * 1.1],
                "n": 30000,
                "p": 122 if with_tensor else 41,
                "deviance": 9800.0,
                "effective_df": 22.0,
                "direct_backend": "gram",
            }
        )
    return {
        "cases": cases,
        "summary": {
            "tensor_multiplier_exact": round(ti_exact / base_exact, 2),
            "tensor_multiplier_discrete": round(ti_discrete / base_discrete, 2),
        },
    }


def _flagship_measurement(median_s: float = 0.9) -> dict:
    steady = [median_s * 0.99, median_s, median_s * 1.01]
    warmup = median_s * 1.1
    return {
        "n_reps": _FLAGSHIP_REPS,
        "warmup_s": warmup,
        "all_times_s": [warmup, *steady],
        "steady_times_s": steady,
        "median_s": median_s,
        "n": 678013,
        "deviance": 212055.4,
        "effective_df": 43.33,
        "direct_backend": "gram",
    }


def _evaluate(gate, baselines, tensor=None, flagship=None, profile=_PROFILE_ID):
    return gate.evaluate_gate(
        baselines,
        tensor=_tensor_measurement() if tensor is None else tensor,
        flagship=_flagship_measurement() if flagship is None else flagship,
        machine_profile=profile,
    )


def _clear_ci_environment(monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("CI", raising=False)


class TestCertificationAndCliOrder:
    def test_matching_enabled_profile_evaluates_thresholds(self, gate, baselines):
        checks = _evaluate(gate, baselines)

        assert checks[0].name == "machine_profile_certification"
        assert checks[0].passed
        assert len(checks) > 1
        assert all(check.passed for check in checks)

    @pytest.mark.parametrize("profile", [None, "", "another-machine"])
    def test_missing_or_mismatched_profile_refuses_before_artifacts(self, gate, baselines, profile):
        checks = gate.evaluate_gate(
            baselines,
            tensor={},
            flagship={},
            machine_profile=profile,
        )

        assert len(checks) == 1
        assert checks[0].name == "machine_profile_certification"
        assert not checks[0].passed

    def test_disabled_profile_refuses_even_when_id_matches(self, gate, baselines):
        baselines["certification_enabled"] = False
        baselines["certification_disabled_reason"] = "fresh calibration required"

        checks = gate.evaluate_gate(
            baselines,
            tensor={"malformed": True},
            flagship={"malformed": True},
            machine_profile=_PROFILE_ID,
        )

        assert len(checks) == 1
        assert not checks[0].passed
        assert "fresh calibration required" in checks[0].detail

    def test_programmatic_profile_argument_cannot_be_omitted(self, gate, baselines):
        with pytest.raises(TypeError, match="machine_profile"):
            gate.evaluate_gate(baselines, tensor={}, flagship={})

    def test_missing_cli_profile_is_nonzero(self, gate, monkeypatch):
        _clear_ci_environment(monkeypatch)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "renamed-local-gate",
                "--baselines",
                "unused.json",
                "--tensor-json",
                "unused.json",
                "--flagship-json",
                "unused.json",
            ],
        )

        with pytest.raises(SystemExit) as exc_info:
            gate.main()

        assert exc_info.value.code != 0

    @pytest.mark.parametrize("enabled", [True, False])
    def test_mismatched_or_disabled_cli_profile_does_not_read_measurements(
        self, gate, baselines, tmp_path, monkeypatch, enabled
    ):
        _clear_ci_environment(monkeypatch)
        baselines["certification_enabled"] = enabled
        if not enabled:
            baselines["certification_disabled_reason"] = "legacy calibration disabled"
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps(baselines))
        profile = "another-machine" if enabled else _PROFILE_ID
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "renamed-local-gate",
                "--machine-profile",
                profile,
                "--baselines",
                str(baseline_path),
                "--tensor-json",
                str(tmp_path / "does-not-exist-tensor.json"),
                "--flagship-json",
                str(tmp_path / "does-not-exist-flagship.json"),
            ],
        )

        assert gate.main() == 1

    @pytest.mark.parametrize(("variable", "value"), [("GITHUB_ACTIONS", "true"), ("CI", "1")])
    def test_ci_runtime_refuses_before_any_file_access(
        self, gate, monkeypatch, capsys, variable, value
    ):
        _clear_ci_environment(monkeypatch)
        monkeypatch.setenv(variable, value)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "/tmp/completely-renamed-entrypoint",
                "--machine-profile",
                _PROFILE_ID,
                "--baselines",
                "/does/not/exist/baseline.json",
                "--tensor-json",
                "/does/not/exist/tensor.json",
                "--flagship-json",
                "/does/not/exist/flagship.json",
            ],
        )

        assert gate.main() == 2
        assert "cannot run" in capsys.readouterr().err


class TestMandatoryCertificationContract:
    @pytest.mark.parametrize(
        ("section", "field", "malformed"),
        [
            ("tensor_cost", "n", "30000"),
            ("tensor_cost", "expected_p", []),
            ("tensor_cost", "expected_outputs", []),
            ("tensor_cost", "output_rtol", "0.005"),
            ("tensor_cost", "expected_backend", 7),
            ("flagship", "n", "678013"),
            ("flagship", "expected", []),
        ],
    )
    @pytest.mark.parametrize("mutation", ["deleted", "malformed"])
    def test_mandatory_baseline_block_cannot_be_deleted_or_malformed(
        self, gate, baselines, section, field, malformed, mutation
    ):
        configured = deepcopy(baselines)
        if mutation == "deleted":
            configured[section].pop(field)
        else:
            configured[section][field] = malformed

        checks = _evaluate(gate, configured)

        assert any(
            check.name.startswith(f"baseline.{section}.{field}") and not check.passed
            for check in checks
        )

    def test_reviewer_timing_only_enabled_profile_is_rejected(self, gate, baselines):
        timing_only_baseline = deepcopy(baselines)
        for field in (
            "n",
            "expected_p",
            "expected_outputs",
            "output_rtol",
            "expected_backend",
        ):
            timing_only_baseline["tensor_cost"].pop(field)
        timing_only_baseline["flagship"].pop("n")
        timing_only_baseline["flagship"].pop("expected")

        tensor = _tensor_measurement()
        for case in tensor["cases"]:
            for field in ("n", "p", "deviance", "effective_df", "direct_backend"):
                case.pop(field)
        flagship = _flagship_measurement()
        for field in ("n", "deviance", "effective_df", "direct_backend"):
            flagship.pop(field)

        checks = _evaluate(
            gate,
            timing_only_baseline,
            tensor=tensor,
            flagship=flagship,
        )

        assert checks[0].passed
        assert any(check.name.startswith("baseline.") and not check.passed for check in checks)
        assert not all(check.passed for check in checks)

    @pytest.mark.parametrize(
        ("artifact", "field"),
        [
            ("tensor", "n"),
            ("tensor", "p"),
            ("tensor", "deviance"),
            ("tensor", "effective_df"),
            ("tensor", "direct_backend"),
            ("flagship", "n"),
            ("flagship", "deviance"),
            ("flagship", "effective_df"),
            ("flagship", "direct_backend"),
        ],
    )
    def test_mandatory_artifact_invariant_cannot_be_omitted(self, gate, baselines, artifact, field):
        tensor = _tensor_measurement()
        flagship = _flagship_measurement()
        if artifact == "tensor":
            tensor["cases"][0].pop(field)
        else:
            flagship.pop(field)

        checks = _evaluate(gate, baselines, tensor=tensor, flagship=flagship)

        assert any(not check.passed for check in checks)
        assert any(field in check.name and not check.passed for check in checks)


class TestTimingAndCompleteness:
    def test_ratio_breach_fails_on_matching_profile(self, gate, baselines):
        checks = _evaluate(gate, baselines, tensor=_tensor_measurement(ti_exact=7.5))

        assert any("tensor_multiplier_exact" in check.name and not check.passed for check in checks)

    def test_complete_fit_breach_fails_when_ratios_stay_clean(self, gate, baselines):
        checks = _evaluate(
            gate,
            baselines,
            tensor=_tensor_measurement(
                base_exact=5.0,
                ti_exact=15.0,
                base_discrete=2.0,
                ti_discrete=5.0,
            ),
        )

        assert any("tensor_cost_base_exact" in check.name and not check.passed for check in checks)

    def test_flagship_complete_fit_breach_fails(self, gate, baselines):
        checks = _evaluate(gate, baselines, flagship=_flagship_measurement(10.0))

        assert any(check.name == "flagship.median_s" and not check.passed for check in checks)

    def test_reviewer_malformed_tensor_reproduction_fails_without_throwing(self, gate, baselines):
        measurement = _tensor_measurement()
        for case in measurement["cases"]:
            case["ok"] = "false"
            case["seconds"] = -1
            case.pop("reps")
            case.pop("rep_seconds")
            case.pop("warmup_seconds")

        checks = _evaluate(gate, baselines, tensor=measurement)

        assert checks
        assert any(not check.passed for check in checks)
        assert any(".ok" in check.name for check in checks)
        assert any(".rep_seconds" in check.name for check in checks)

    @pytest.mark.parametrize(
        "tag",
        [
            "tensor_cost_base_exact",
            "tensor_cost_base_discrete",
            "tensor_cost_ti_exact",
            "tensor_cost_ti_discrete",
        ],
    )
    @pytest.mark.parametrize("field", ["tensor", "discrete"])
    def test_wrong_workload_boolean_for_tag_fails(self, gate, baselines, tag, field):
        measurement = _tensor_measurement()
        case = next(case for case in measurement["cases"] if case["tag"] == tag)
        case[field] = not case[field]

        checks = _evaluate(gate, baselines, tensor=measurement)

        assert any(
            check.name == f"tensor_cost.{tag}.{field}" and not check.passed for check in checks
        )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("tensor", 1),
            ("tensor", "false"),
            ("discrete", 0),
            ("discrete", "true"),
        ],
    )
    def test_truthy_or_string_workload_flags_fail(self, gate, baselines, field, value):
        measurement = _tensor_measurement()
        measurement["cases"][0][field] = value

        checks = _evaluate(gate, baselines, tensor=measurement)

        assert any(
            check.name == f"tensor_cost.tensor_cost_base_exact.{field}" and not check.passed
            for check in checks
        )

    @pytest.mark.parametrize("field", ["tensor", "discrete"])
    def test_missing_workload_flags_fail(self, gate, baselines, field):
        measurement = _tensor_measurement()
        measurement["cases"][0].pop(field)

        checks = _evaluate(gate, baselines, tensor=measurement)

        assert any(
            check.name == f"tensor_cost.tensor_cost_base_exact.{field}" and not check.passed
            for check in checks
        )

    @pytest.mark.parametrize(
        "mutation",
        [
            "missing_case",
            "duplicate_case",
            "ok_string",
            "ok_integer",
            "missing_reps",
            "reps_bool",
            "reps_string",
            "wrong_rep_count",
            "rep_nan",
            "rep_inf",
            "rep_string",
            "rep_bool",
            "rep_negative",
            "missing_warmup",
            "wrong_warmup_count",
            "seconds_nan",
            "seconds_inf",
            "seconds_string",
            "seconds_bool",
            "seconds_negative",
            "seconds_inconsistent",
            "missing_ratio",
            "ratio_nan",
            "ratio_inf",
            "ratio_string",
            "ratio_bool",
            "ratio_inconsistent",
        ],
    )
    def test_malformed_tensor_protocol_fails_closed(self, gate, baselines, mutation):
        measurement = _tensor_measurement()
        case = measurement["cases"][0]
        if mutation == "missing_case":
            measurement["cases"].pop()
        elif mutation == "duplicate_case":
            measurement["cases"][-1]["tag"] = measurement["cases"][0]["tag"]
        elif mutation == "ok_string":
            case["ok"] = "false"
        elif mutation == "ok_integer":
            case["ok"] = 1
        elif mutation == "missing_reps":
            case.pop("reps")
        elif mutation == "reps_bool":
            case["reps"] = True
        elif mutation == "reps_string":
            case["reps"] = str(_TENSOR_REPS)
        elif mutation == "wrong_rep_count":
            case["rep_seconds"].pop()
        elif mutation == "rep_nan":
            case["rep_seconds"][0] = float("nan")
        elif mutation == "rep_inf":
            case["rep_seconds"][0] = float("inf")
        elif mutation == "rep_string":
            case["rep_seconds"][0] = "0.5"
        elif mutation == "rep_bool":
            case["rep_seconds"][0] = True
        elif mutation == "rep_negative":
            case["rep_seconds"][0] = -0.5
        elif mutation == "missing_warmup":
            case.pop("warmup_seconds")
        elif mutation == "wrong_warmup_count":
            case["warmup_seconds"] = []
        elif mutation == "seconds_nan":
            case["seconds"] = float("nan")
        elif mutation == "seconds_inf":
            case["seconds"] = float("inf")
        elif mutation == "seconds_string":
            case["seconds"] = "0.5"
        elif mutation == "seconds_bool":
            case["seconds"] = True
        elif mutation == "seconds_negative":
            case["seconds"] = -0.5
        elif mutation == "seconds_inconsistent":
            case["seconds"] = 0.7
        elif mutation == "missing_ratio":
            measurement["summary"].pop("tensor_multiplier_exact")
        elif mutation == "ratio_nan":
            measurement["summary"]["tensor_multiplier_exact"] = float("nan")
        elif mutation == "ratio_inf":
            measurement["summary"]["tensor_multiplier_exact"] = float("inf")
        elif mutation == "ratio_string":
            measurement["summary"]["tensor_multiplier_exact"] = "3.0"
        elif mutation == "ratio_bool":
            measurement["summary"]["tensor_multiplier_exact"] = True
        else:
            measurement["summary"]["tensor_multiplier_exact"] = 4.0

        checks = _evaluate(gate, baselines, tensor=measurement)

        assert any(not check.passed for check in checks), mutation

    @pytest.mark.parametrize(
        "mutation",
        [
            "n_reps_missing",
            "n_reps_bool",
            "n_reps_string",
            "all_missing",
            "all_count",
            "all_nan",
            "all_inf",
            "all_string",
            "all_bool",
            "all_negative",
            "steady_missing",
            "steady_count",
            "steady_mismatch",
            "warmup_missing",
            "warmup_inconsistent",
            "median_nan",
            "median_inf",
            "median_string",
            "median_bool",
            "median_negative",
            "median_inconsistent",
        ],
    )
    def test_malformed_flagship_protocol_fails_closed(self, gate, baselines, mutation):
        flagship = _flagship_measurement()
        if mutation == "n_reps_missing":
            flagship.pop("n_reps")
        elif mutation == "n_reps_bool":
            flagship["n_reps"] = True
        elif mutation == "n_reps_string":
            flagship["n_reps"] = str(_FLAGSHIP_REPS)
        elif mutation == "all_missing":
            flagship.pop("all_times_s")
        elif mutation == "all_count":
            flagship["all_times_s"].pop()
        elif mutation == "all_nan":
            flagship["all_times_s"][0] = float("nan")
        elif mutation == "all_inf":
            flagship["all_times_s"][0] = float("inf")
        elif mutation == "all_string":
            flagship["all_times_s"][0] = "0.9"
        elif mutation == "all_bool":
            flagship["all_times_s"][0] = True
        elif mutation == "all_negative":
            flagship["all_times_s"][0] = -0.9
        elif mutation == "steady_missing":
            flagship.pop("steady_times_s")
        elif mutation == "steady_count":
            flagship["steady_times_s"].pop()
        elif mutation == "steady_mismatch":
            flagship["steady_times_s"][0] *= 1.5
        elif mutation == "warmup_missing":
            flagship.pop("warmup_s")
        elif mutation == "warmup_inconsistent":
            flagship["warmup_s"] *= 1.5
        elif mutation == "median_nan":
            flagship["median_s"] = float("nan")
        elif mutation == "median_inf":
            flagship["median_s"] = float("inf")
        elif mutation == "median_string":
            flagship["median_s"] = "0.9"
        elif mutation == "median_bool":
            flagship["median_s"] = True
        elif mutation == "median_negative":
            flagship["median_s"] = -0.9
        else:
            flagship["median_s"] = 1.2

        checks = _evaluate(gate, baselines, flagship=flagship)

        assert any(not check.passed for check in checks), mutation


@pytest.fixture()
def invariant_baselines(baselines):
    configured = deepcopy(baselines)
    configured["tensor_cost"].update(
        n=30000,
        expected_p={
            "tensor_cost_base_exact": 41,
            "tensor_cost_base_discrete": 41,
            "tensor_cost_ti_exact": 122,
            "tensor_cost_ti_discrete": 122,
        },
        output_rtol=5e-3,
        expected_outputs={
            tag: {"deviance": 9800.0, "effective_df": 22.0}
            for tag in (
                "tensor_cost_base_exact",
                "tensor_cost_base_discrete",
                "tensor_cost_ti_exact",
                "tensor_cost_ti_discrete",
            )
        },
        expected_backend="gram",
    )
    configured["flagship"].update(
        n=678013,
        expected={
            "deviance": 212055.4,
            "effective_df": 43.33,
            "rtol": 1e-3,
            "backend": "gram",
        },
    )
    return configured


def _measurement_with_invariants(deviance: float = 9800.0, backend: str = "gram") -> dict:
    measurement = _tensor_measurement()
    for case in measurement["cases"]:
        case.update(
            n=30000,
            p=122 if "_ti_" in case["tag"] else 41,
            deviance=deviance,
            effective_df=22.0,
            direct_backend=backend,
        )
    return measurement


def _flagship_with_invariants(
    *,
    deviance: float = 212055.4,
    effective_df: float = 43.33,
    backend: str = "gram",
    n: int = 678013,
) -> dict:
    flagship = _flagship_measurement()
    flagship.update(
        deviance=deviance,
        effective_df=effective_df,
        direct_backend=backend,
        n=n,
    )
    return flagship


class TestNumericalDimensionAndBackendInvariants:
    def test_matching_invariants_pass(self, gate, invariant_baselines):
        checks = _evaluate(
            gate,
            invariant_baselines,
            tensor=_measurement_with_invariants(),
            flagship=_flagship_with_invariants(),
        )

        assert all(check.passed for check in checks)

    @pytest.mark.parametrize(
        ("mutation", "expected_name"),
        [
            ("tensor_deviance", "tensor_cost.tensor_cost_base_exact.deviance.rel_dev"),
            ("tensor_width", "tensor_cost.tensor_cost_ti_exact.p"),
            ("tensor_rows", "tensor_cost.tensor_cost_base_exact.n"),
            ("tensor_backend", "tensor_cost.tensor_cost_base_exact.backend"),
            ("flagship_deviance", "flagship.deviance.rel_dev"),
            ("flagship_edf", "flagship.effective_df.rel_dev"),
            ("flagship_rows", "flagship.n"),
            ("flagship_backend", "flagship.backend"),
        ],
    )
    def test_invariant_failure_is_independent_of_timing(
        self, gate, invariant_baselines, mutation, expected_name
    ):
        tensor = _measurement_with_invariants()
        flagship = _flagship_with_invariants()
        if mutation == "tensor_deviance":
            tensor["cases"][0]["deviance"] = 9900.0
        elif mutation == "tensor_width":
            tensor["cases"][2]["p"] = 41
        elif mutation == "tensor_rows":
            tensor["cases"][0]["n"] = 5000
        elif mutation == "tensor_backend":
            tensor["cases"][0]["direct_backend"] = "qr"
        elif mutation == "flagship_deviance":
            flagship["deviance"] = 213000.0
        elif mutation == "flagship_edf":
            flagship["effective_df"] = 39.0
        elif mutation == "flagship_rows":
            flagship["n"] = 600000
        else:
            flagship["direct_backend"] = "structured"

        checks = _evaluate(
            gate,
            invariant_baselines,
            tensor=tensor,
            flagship=flagship,
        )

        assert any(check.name == expected_name and not check.passed for check in checks)
        timing_checks = [
            check
            for check in checks
            if check.name.endswith("median_s") or "tensor_multiplier" in check.name
        ]
        assert timing_checks
        assert all(check.passed for check in timing_checks)

    @pytest.mark.parametrize(
        ("artifact", "field", "value"),
        [
            ("tensor", "p", True),
            ("tensor", "p", "41"),
            ("tensor", "n", 30000.0),
            ("tensor", "n", None),
            ("tensor", "deviance", float("nan")),
            ("tensor", "effective_df", float("inf")),
            ("tensor", "deviance", "9800"),
            ("tensor", "effective_df", False),
            ("flagship", "n", True),
            ("flagship", "n", "678013"),
            ("flagship", "deviance", float("nan")),
            ("flagship", "effective_df", float("inf")),
            ("flagship", "deviance", "212055.4"),
            ("flagship", "effective_df", False),
        ],
    )
    def test_malformed_invariant_values_fail_closed(
        self, gate, invariant_baselines, artifact, field, value
    ):
        tensor = _measurement_with_invariants()
        flagship = _flagship_with_invariants()
        if artifact == "tensor":
            tensor["cases"][0][field] = value
        else:
            flagship[field] = value

        checks = _evaluate(
            gate,
            invariant_baselines,
            tensor=tensor,
            flagship=flagship,
        )

        assert any(not check.passed for check in checks)


class TestCommittedConfiguration:
    def test_committed_legacy_baseline_is_explicitly_non_certifying(self, gate):
        committed = json.loads(_BASELINE_PATH.read_text())

        assert committed["machine_profile_id"] == "superglm-reference-box-2026-07-29"
        assert committed["certification_enabled"] is False
        assert committed["certification_disabled_reason"]
        checks = gate.evaluate_gate(
            committed,
            tensor={"would": "otherwise be malformed"},
            flagship={"would": "otherwise be malformed"},
            machine_profile=committed["machine_profile_id"],
        )
        assert len(checks) == 1
        assert not checks[0].passed

    def test_committed_historical_provenance_is_complete(self):
        committed = json.loads(_BASELINE_PATH.read_text())
        tensor = committed["tensor_cost"]
        for tag in (
            "tensor_cost_base_exact",
            "tensor_cost_base_discrete",
            "tensor_cost_ti_exact",
            "tensor_cost_ti_discrete",
        ):
            assert tensor["reference_seconds"][tag] > 0
            assert tensor["expected_p"][tag] > 0
            assert tensor["expected_outputs"][tag]["deviance"] > 0
            assert tensor["expected_outputs"][tag]["effective_df"] > 0
        assert tensor["n"] == 30000
        assert tensor["reps"] == 5
        assert tensor["checks"]["tensor_multiplier_exact"]["max"] > 0
        assert tensor["checks"]["tensor_multiplier_discrete"]["max"] > 0
        assert tensor["checks"]["absolute_multiple"]["max"] > 0
        assert tensor["output_rtol"] > 0
        assert tensor["expected_backend"]
        flagship = committed["flagship"]
        assert flagship["n"] == 678013
        assert flagship["reps"] == 30
        assert flagship["reference_median_s"] > 0
        assert flagship["checks"]["absolute_multiple"]["max"] > 0
        assert flagship["expected"]["deviance"] > 0
        assert flagship["expected"]["effective_df"] > 0
        assert flagship["expected"]["rtol"] > 0
        assert flagship["expected"]["backend"]

    def test_old_ci_named_gate_and_baseline_are_gone(self):
        assert not (_ROOT / "benchmarks" / "ci_perf_gate.py").exists()
        assert not (_ROOT / "benchmarks" / "results" / "ci_perf_baselines.json").exists()

    def test_hosted_ci_has_no_wall_time_or_rss_gate(self):
        workflow_dir = _ROOT / ".github" / "workflows"
        workflow_paths = sorted((*workflow_dir.glob("*.yml"), *workflow_dir.glob("*.yaml")))
        assert workflow_paths
        for forbidden in (
            "perf-gate:",
            "benchmark_tensor_cost",
            "timing_30rep_superglm",
            "local_perf_gate",
            "ci_perf_gate",
            "perf-gate-measurements",
            "peak RSS",
        ):
            for workflow_path in workflow_paths:
                assert forbidden not in workflow_path.read_text(), workflow_path
