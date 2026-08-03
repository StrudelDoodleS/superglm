"""Local-only performance gate for an explicitly certified machine profile.

Wall-time certification is available only when the baseline both enables
certification and names the operator-asserted local profile.  The profile ID is
not a hardware fingerprint.  Hosted CI is refused before any artifact is read.

The committed historical baseline is intentionally non-certifying.  A future
operator must establish and freshly calibrate a stable local machine,
software, and thread environment before enabling a new profile.

Once enabled, the gate checks complete-fit wall time, within-run tensor/base
ratios, numerical outputs, dimensions, row counts, backend dispatch, and the
benchmark repetition protocol.  Malformed or incomplete artifacts fail closed.

Usage after fresh calibration and explicit baseline enablement::

    uv run python benchmarks/local_perf_gate.py \
        --machine-profile "$LOCAL_PERF_PROFILE" \
        --baselines benchmarks/results/local_perf_baselines.json \
        --tensor-json /tmp/tensor_cost_local.json \
        --flagship-json /tmp/flagship_local.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

TENSOR_CASE_TAGS = (
    "tensor_cost_base_exact",
    "tensor_cost_base_discrete",
    "tensor_cost_ti_exact",
    "tensor_cost_ti_discrete",
)
TENSOR_RATIO_CASES = {
    "tensor_multiplier_exact": ("tensor_cost_ti_exact", "tensor_cost_base_exact"),
    "tensor_multiplier_discrete": (
        "tensor_cost_ti_discrete",
        "tensor_cost_base_discrete",
    ),
}
TENSOR_CASE_WORKLOAD = {
    "tensor_cost_base_exact": (False, False),
    "tensor_cost_base_discrete": (False, True),
    "tensor_cost_ti_exact": (True, False),
    "tensor_cost_ti_discrete": (True, True),
}
_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})


@dataclass(frozen=True)
class GateCheck:
    name: str
    measured: float | None
    limit: float
    passed: bool
    detail: str | None = None

    def render(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        measured = "missing" if self.measured is None else f"{self.measured:.3f}"
        suffix = f" ({self.detail})" if self.detail else ""
        return f"  [{status}] {self.name}: measured {measured}, limit {self.limit:.3f}{suffix}"


@dataclass(frozen=True)
class TensorConfig:
    reps: int
    reference_seconds: dict[str, float]
    ratio_limits: dict[str, float]
    absolute_multiple: float
    expected_n: int
    expected_p: dict[str, int]
    expected_outputs: dict[str, dict[str, float]]
    output_rtol: float
    expected_backend: str


@dataclass(frozen=True)
class FlagshipConfig:
    reps: int
    reference_median_s: float
    absolute_multiple: float
    expected_n: int
    expected: dict[str, float | str]


@dataclass(frozen=True)
class TensorArtifact:
    cases: dict[str, dict]
    medians: dict[str, float]
    ratios: dict[str, float]


@dataclass(frozen=True)
class FlagshipArtifact:
    payload: dict
    median_s: float


def _failure(name: str, detail: str) -> GateCheck:
    return GateCheck(name=name, measured=None, limit=0.0, passed=False, detail=detail)


def _finite_real(value: object, *, positive: bool = False) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    measured = float(value)
    if not math.isfinite(measured) or (positive and measured <= 0.0):
        return None
    return measured


def _nonnegative_real(value: object) -> float | None:
    measured = _finite_real(value)
    return measured if measured is not None and measured >= 0.0 else None


def _exact_int(value: object, *, positive: bool = False) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if positive and value <= 0:
        return None
    return value


def _mapping(value: object) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _check(
    name: str,
    measured: object,
    limit: object,
    detail: str | None = None,
) -> GateCheck:
    measured_number = _finite_real(measured)
    limit_number = _nonnegative_real(limit)
    passed = (
        measured_number is not None and limit_number is not None and measured_number <= limit_number
    )
    return GateCheck(
        name=name,
        measured=measured_number,
        limit=0.0 if limit_number is None else limit_number,
        passed=passed,
        detail=detail,
    )


def certification_check(baselines: object, machine_profile: str | None) -> GateCheck:
    """Validate the operator profile assertion and explicit certification state."""
    baseline_map = _mapping(baselines)
    expected = baseline_map.get("machine_profile_id")
    enabled = baseline_map.get("certification_enabled")
    reason = baseline_map.get("certification_disabled_reason")
    profile_matches = (
        isinstance(expected, str)
        and bool(expected)
        and isinstance(machine_profile, str)
        and bool(machine_profile)
        and machine_profile == expected
    )
    passed = profile_matches and enabled is True
    detail = (
        f"expected profile {expected!r}, got {machine_profile!r}; certification_enabled={enabled!r}"
    )
    if enabled is not True and isinstance(reason, str) and reason:
        detail += f"; reason: {reason}"
    return GateCheck(
        name="machine_profile_certification",
        measured=0.0 if passed else None,
        limit=0.0,
        passed=passed,
        detail=detail,
    )


def _positive_mapping(
    value: object,
    keys: Sequence[str],
    *,
    path: str,
    failures: list[GateCheck],
) -> dict[str, float]:
    source = _mapping(value)
    result: dict[str, float] = {}
    if not isinstance(value, Mapping):
        failures.append(_failure(path, "expected an object"))
    for key in keys:
        measured = _finite_real(source.get(key), positive=True)
        if measured is None:
            failures.append(_failure(f"{path}.{key}", "expected a finite positive number"))
        else:
            result[key] = measured
    return result


def _baseline_config(
    baselines: object,
) -> tuple[TensorConfig | None, FlagshipConfig | None, list[GateCheck]]:
    failures: list[GateCheck] = []
    baseline_map = _mapping(baselines)
    tensor = _mapping(baseline_map.get("tensor_cost"))
    flagship = _mapping(baseline_map.get("flagship"))
    if not isinstance(baseline_map.get("tensor_cost"), Mapping):
        failures.append(_failure("baseline.tensor_cost", "expected an object"))
    if not isinstance(baseline_map.get("flagship"), Mapping):
        failures.append(_failure("baseline.flagship", "expected an object"))

    tensor_reps = _exact_int(tensor.get("reps"), positive=True)
    if tensor_reps is None:
        failures.append(_failure("baseline.tensor_cost.reps", "expected a positive integer"))
    reference_seconds = _positive_mapping(
        tensor.get("reference_seconds"),
        TENSOR_CASE_TAGS,
        path="baseline.tensor_cost.reference_seconds",
        failures=failures,
    )
    tensor_checks = _mapping(tensor.get("checks"))
    ratio_limits: dict[str, float] = {}
    for ratio_name in TENSOR_RATIO_CASES:
        limit = _finite_real(
            _mapping(tensor_checks.get(ratio_name)).get("max"),
            positive=True,
        )
        if limit is None:
            failures.append(
                _failure(
                    f"baseline.tensor_cost.checks.{ratio_name}.max",
                    "expected a finite positive number",
                )
            )
        else:
            ratio_limits[ratio_name] = limit
    tensor_multiple = _finite_real(
        _mapping(tensor_checks.get("absolute_multiple")).get("max"),
        positive=True,
    )
    if tensor_multiple is None:
        failures.append(
            _failure(
                "baseline.tensor_cost.checks.absolute_multiple.max",
                "expected a finite positive number",
            )
        )

    expected_n = _exact_int(tensor.get("n"), positive=True)
    if expected_n is None:
        failures.append(_failure("baseline.tensor_cost.n", "expected a positive integer"))

    expected_p: dict[str, int] = {}
    expected_p_source = _mapping(tensor.get("expected_p"))
    if not isinstance(tensor.get("expected_p"), Mapping):
        failures.append(_failure("baseline.tensor_cost.expected_p", "expected an object"))
    for tag in TENSOR_CASE_TAGS:
        width = _exact_int(expected_p_source.get(tag), positive=True)
        if width is None:
            failures.append(
                _failure(
                    f"baseline.tensor_cost.expected_p.{tag}",
                    "expected a positive integer",
                )
            )
        else:
            expected_p[tag] = width

    expected_outputs: dict[str, dict[str, float]] = {}
    output_source = _mapping(tensor.get("expected_outputs"))
    if not isinstance(tensor.get("expected_outputs"), Mapping):
        failures.append(_failure("baseline.tensor_cost.expected_outputs", "expected an object"))
    for tag in TENSOR_CASE_TAGS:
        tag_outputs = _mapping(output_source.get(tag))
        expected_outputs[tag] = {}
        for field in ("deviance", "effective_df"):
            expected_value = _finite_real(tag_outputs.get(field), positive=True)
            if expected_value is None:
                failures.append(
                    _failure(
                        f"baseline.tensor_cost.expected_outputs.{tag}.{field}",
                        "expected a finite positive number",
                    )
                )
            else:
                expected_outputs[tag][field] = expected_value
    output_rtol = _nonnegative_real(tensor.get("output_rtol"))
    if output_rtol is None:
        failures.append(
            _failure(
                "baseline.tensor_cost.output_rtol",
                "expected a finite non-negative number",
            )
        )

    expected_backend: str | None = None
    backend = tensor.get("expected_backend")
    if not isinstance(backend, str) or not backend:
        failures.append(
            _failure("baseline.tensor_cost.expected_backend", "expected a non-empty string")
        )
    else:
        expected_backend = backend

    flagship_reps = _exact_int(flagship.get("reps"), positive=True)
    if flagship_reps is None or flagship_reps < 2:
        failures.append(_failure("baseline.flagship.reps", "expected an integer of at least two"))
    flagship_reference = _finite_real(flagship.get("reference_median_s"), positive=True)
    if flagship_reference is None:
        failures.append(
            _failure(
                "baseline.flagship.reference_median_s",
                "expected a finite positive number",
            )
        )
    flagship_multiple = _finite_real(
        _mapping(_mapping(flagship.get("checks")).get("absolute_multiple")).get("max"),
        positive=True,
    )
    if flagship_multiple is None:
        failures.append(
            _failure(
                "baseline.flagship.checks.absolute_multiple.max",
                "expected a finite positive number",
            )
        )

    flagship_n = _exact_int(flagship.get("n"), positive=True)
    if flagship_n is None:
        failures.append(_failure("baseline.flagship.n", "expected a positive integer"))

    expected_source = _mapping(flagship.get("expected"))
    if not isinstance(flagship.get("expected"), Mapping):
        failures.append(_failure("baseline.flagship.expected", "expected an object"))
    flagship_expected: dict[str, float | str] = {}
    for field in ("deviance", "effective_df"):
        expected_value = _finite_real(expected_source.get(field), positive=True)
        if expected_value is None:
            failures.append(
                _failure(
                    f"baseline.flagship.expected.{field}",
                    "expected a finite positive number",
                )
            )
        else:
            flagship_expected[field] = expected_value
    rtol = _nonnegative_real(expected_source.get("rtol"))
    if rtol is None:
        failures.append(
            _failure(
                "baseline.flagship.expected.rtol",
                "expected a finite non-negative number",
            )
        )
    else:
        flagship_expected["rtol"] = rtol
    flagship_backend = expected_source.get("backend")
    if not isinstance(flagship_backend, str) or not flagship_backend:
        failures.append(
            _failure(
                "baseline.flagship.expected.backend",
                "expected a non-empty string",
            )
        )
    else:
        flagship_expected["backend"] = flagship_backend

    if failures:
        return None, None, failures
    assert tensor_reps is not None
    assert tensor_multiple is not None
    assert expected_n is not None
    assert output_rtol is not None
    assert expected_backend is not None
    assert flagship_reps is not None
    assert flagship_reference is not None
    assert flagship_multiple is not None
    assert flagship_n is not None
    return (
        TensorConfig(
            reps=tensor_reps,
            reference_seconds=reference_seconds,
            ratio_limits=ratio_limits,
            absolute_multiple=tensor_multiple,
            expected_n=expected_n,
            expected_p=expected_p,
            expected_outputs=expected_outputs,
            output_rtol=output_rtol,
            expected_backend=expected_backend,
        ),
        FlagshipConfig(
            reps=flagship_reps,
            reference_median_s=flagship_reference,
            absolute_multiple=flagship_multiple,
            expected_n=flagship_n,
            expected=flagship_expected,
        ),
        [],
    )


def _positive_real_list(
    value: object,
    *,
    expected_length: int,
    path: str,
    failures: list[GateCheck],
) -> list[float] | None:
    if not isinstance(value, list):
        failures.append(_failure(path, "expected a list"))
        return None
    if len(value) != expected_length:
        failures.append(_failure(path, f"expected {expected_length} samples, got {len(value)}"))
    values: list[float] = []
    for index, item in enumerate(value):
        measured = _finite_real(item, positive=True)
        if measured is None:
            failures.append(_failure(f"{path}[{index}]", "expected a finite positive number"))
        else:
            values.append(measured)
    return values if len(values) == expected_length and len(value) == expected_length else None


def _tensor_artifact(
    tensor: object,
    config: TensorConfig,
) -> tuple[TensorArtifact | None, list[GateCheck]]:
    failures: list[GateCheck] = []
    tensor_map = _mapping(tensor)
    if not isinstance(tensor, Mapping):
        return None, [_failure("tensor_cost.artifact", "expected an object")]
    cases_value = tensor_map.get("cases")
    if not isinstance(cases_value, list):
        return None, [_failure("tensor_cost.cases", "expected a list")]
    if len(cases_value) != len(TENSOR_CASE_TAGS):
        failures.append(
            _failure(
                "tensor_cost.cases",
                f"expected exactly {len(TENSOR_CASE_TAGS)} cases, got {len(cases_value)}",
            )
        )

    cases: dict[str, dict] = {}
    observed_tags: list[str] = []
    for index, case_value in enumerate(cases_value):
        if not isinstance(case_value, Mapping):
            failures.append(_failure(f"tensor_cost.cases[{index}]", "expected an object"))
            continue
        case = dict(case_value)
        tag = case.get("tag")
        if not isinstance(tag, str):
            failures.append(_failure(f"tensor_cost.cases[{index}].tag", "expected a string"))
            continue
        observed_tags.append(tag)
        if tag in cases:
            failures.append(_failure("tensor_cost.cases", f"duplicate case tag {tag!r}"))
        else:
            cases[tag] = case
    missing = sorted(set(TENSOR_CASE_TAGS) - set(observed_tags))
    unexpected = sorted(set(observed_tags) - set(TENSOR_CASE_TAGS))
    if missing:
        failures.append(_failure("tensor_cost.cases", f"missing case tags: {missing}"))
    if unexpected:
        failures.append(_failure("tensor_cost.cases", f"unexpected case tags: {unexpected}"))

    medians: dict[str, float] = {}
    expected_warmups = 1 if config.reps >= 2 else 0
    for tag in TENSOR_CASE_TAGS:
        case = cases.get(tag)
        if case is None:
            continue
        path = f"tensor_cost.{tag}"
        if case.get("ok") is not True:
            failures.append(_failure(f"{path}.ok", "expected the boolean true"))
        expected_tensor, expected_discrete = TENSOR_CASE_WORKLOAD[tag]
        for field, expected_value in (
            ("tensor", expected_tensor),
            ("discrete", expected_discrete),
        ):
            observed_value = case.get(field)
            if observed_value is not expected_value:
                failures.append(
                    _failure(
                        f"{path}.{field}",
                        f"expected the boolean {expected_value!r}, got {observed_value!r}",
                    )
                )
        case_reps = _exact_int(case.get("reps"), positive=True)
        if case_reps != config.reps:
            failures.append(
                _failure(
                    f"{path}.reps", f"expected integer {config.reps}, got {case.get('reps')!r}"
                )
            )
        rep_seconds = _positive_real_list(
            case.get("rep_seconds"),
            expected_length=config.reps,
            path=f"{path}.rep_seconds",
            failures=failures,
        )
        _positive_real_list(
            case.get("warmup_seconds"),
            expected_length=expected_warmups,
            path=f"{path}.warmup_seconds",
            failures=failures,
        )
        reported_median = _finite_real(case.get("seconds"), positive=True)
        if reported_median is None:
            failures.append(_failure(f"{path}.seconds", "expected a finite positive number"))
        elif rep_seconds is not None:
            sample_median = float(statistics.median(rep_seconds))
            if not math.isclose(
                reported_median,
                sample_median,
                rel_tol=0.0,
                abs_tol=0.001000001,
            ):
                failures.append(
                    _failure(
                        f"{path}.seconds",
                        f"reported {reported_median} is inconsistent with sample median "
                        f"{sample_median}",
                    )
                )
            else:
                medians[tag] = reported_median

        if config.expected_p is not None and _exact_int(case.get("p"), positive=True) is None:
            failures.append(_failure(f"{path}.p", "expected a positive integer"))
        if config.expected_n is not None and _exact_int(case.get("n"), positive=True) is None:
            failures.append(_failure(f"{path}.n", "expected a positive integer"))
        if config.expected_outputs is not None:
            for field in ("deviance", "effective_df"):
                if _finite_real(case.get(field)) is None:
                    failures.append(_failure(f"{path}.{field}", "expected a finite real number"))
        if config.expected_backend is not None:
            backend = case.get("direct_backend")
            if not isinstance(backend, str) or not backend:
                failures.append(_failure(f"{path}.direct_backend", "expected a non-empty string"))

    summary = _mapping(tensor_map.get("summary"))
    if not isinstance(tensor_map.get("summary"), Mapping):
        failures.append(_failure("tensor_cost.summary", "expected an object"))
    ratios: dict[str, float] = {}
    for ratio_name, (numerator_tag, denominator_tag) in TENSOR_RATIO_CASES.items():
        summary_ratio = _finite_real(summary.get(ratio_name), positive=True)
        if summary_ratio is None:
            failures.append(
                _failure(
                    f"tensor_cost.summary.{ratio_name}",
                    "expected a finite positive number",
                )
            )
            continue
        if numerator_tag not in medians or denominator_tag not in medians:
            continue
        derived_ratio = medians[numerator_tag] / medians[denominator_tag]
        producer_ratio = round(derived_ratio, 2)
        if not math.isclose(summary_ratio, producer_ratio, rel_tol=0.0, abs_tol=1e-12):
            failures.append(
                _failure(
                    f"tensor_cost.summary.{ratio_name}",
                    f"reported {summary_ratio} is inconsistent with case medians "
                    f"(expected {producer_ratio})",
                )
            )
        else:
            ratios[ratio_name] = derived_ratio

    if failures:
        return None, failures
    return TensorArtifact(cases=cases, medians=medians, ratios=ratios), []


def _same_samples(left: Sequence[float], right: Sequence[float]) -> bool:
    return len(left) == len(right) and all(
        math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(left, right, strict=True)
    )


def _flagship_artifact(
    flagship: object,
    config: FlagshipConfig,
) -> tuple[FlagshipArtifact | None, list[GateCheck]]:
    failures: list[GateCheck] = []
    if not isinstance(flagship, Mapping):
        return None, [_failure("flagship.artifact", "expected an object")]
    payload = dict(flagship)
    n_reps = _exact_int(payload.get("n_reps"), positive=True)
    if n_reps != config.reps:
        failures.append(
            _failure(
                "flagship.n_reps",
                f"expected integer {config.reps}, got {payload.get('n_reps')!r}",
            )
        )
    all_times = _positive_real_list(
        payload.get("all_times_s"),
        expected_length=config.reps,
        path="flagship.all_times_s",
        failures=failures,
    )
    steady_times = _positive_real_list(
        payload.get("steady_times_s"),
        expected_length=config.reps - 1,
        path="flagship.steady_times_s",
        failures=failures,
    )
    warmup = _finite_real(payload.get("warmup_s"), positive=True)
    if warmup is None:
        failures.append(_failure("flagship.warmup_s", "expected a finite positive number"))
    elif all_times is not None and not math.isclose(
        warmup,
        all_times[0],
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        failures.append(
            _failure("flagship.warmup_s", "does not match the first all_times_s sample")
        )
    if (
        all_times is not None
        and steady_times is not None
        and not _same_samples(steady_times, all_times[1:])
    ):
        failures.append(
            _failure("flagship.steady_times_s", "does not match all_times_s after warmup")
        )

    median_s = _finite_real(payload.get("median_s"), positive=True)
    if median_s is None:
        failures.append(_failure("flagship.median_s", "expected a finite positive number"))
    elif steady_times is not None:
        sample_median = float(statistics.median(steady_times))
        if not math.isclose(median_s, sample_median, rel_tol=1e-12, abs_tol=1e-12):
            failures.append(
                _failure(
                    "flagship.median_s",
                    f"reported {median_s} does not match sample median {sample_median}",
                )
            )

    if config.expected_n is not None and _exact_int(payload.get("n"), positive=True) is None:
        failures.append(_failure("flagship.n", "expected a positive integer"))
    if config.expected is not None:
        for field in ("deviance", "effective_df"):
            if _finite_real(payload.get(field)) is None:
                failures.append(_failure(f"flagship.{field}", "expected a finite real number"))
        if "backend" in config.expected:
            backend = payload.get("direct_backend")
            if not isinstance(backend, str) or not backend:
                failures.append(_failure("flagship.direct_backend", "expected a non-empty string"))

    if failures or median_s is None:
        return None, failures
    return FlagshipArtifact(payload=payload, median_s=median_s), []


def _evaluate_certified(
    baselines: object,
    tensor: object,
    flagship: object,
    certification: GateCheck,
) -> list[GateCheck]:
    checks = [certification]
    tensor_config, flagship_config, config_failures = _baseline_config(baselines)
    if config_failures:
        return checks + config_failures
    assert tensor_config is not None
    assert flagship_config is not None

    tensor_artifact, tensor_failures = _tensor_artifact(tensor, tensor_config)
    flagship_artifact, flagship_failures = _flagship_artifact(flagship, flagship_config)
    if tensor_failures or flagship_failures:
        return checks + tensor_failures + flagship_failures
    assert tensor_artifact is not None
    assert flagship_artifact is not None

    for ratio_name in TENSOR_RATIO_CASES:
        checks.append(
            _check(
                f"tensor_cost.{ratio_name}",
                tensor_artifact.ratios[ratio_name],
                tensor_config.ratio_limits[ratio_name],
            )
        )
    for tag in TENSOR_CASE_TAGS:
        case = tensor_artifact.cases[tag]
        checks.append(
            _check(
                f"tensor_cost.{tag}.median_s",
                tensor_artifact.medians[tag],
                tensor_config.reference_seconds[tag] * tensor_config.absolute_multiple,
            )
        )
        if tensor_config.expected_p is not None:
            observed_p = _exact_int(case.get("p"), positive=True)
            checks.append(
                _check(
                    f"tensor_cost.{tag}.p",
                    None if observed_p is None else abs(observed_p - tensor_config.expected_p[tag]),
                    0.0,
                )
            )
        if tensor_config.expected_n is not None:
            observed_n = _exact_int(case.get("n"), positive=True)
            checks.append(
                _check(
                    f"tensor_cost.{tag}.n",
                    None if observed_n is None else abs(observed_n - tensor_config.expected_n),
                    0.0,
                )
            )
        if tensor_config.expected_outputs is not None:
            assert tensor_config.output_rtol is not None
            for field in ("deviance", "effective_df"):
                measured_value = _finite_real(case.get(field))
                expected_value = tensor_config.expected_outputs[tag][field]
                deviation = (
                    None if measured_value is None else abs(measured_value / expected_value - 1.0)
                )
                checks.append(
                    _check(
                        f"tensor_cost.{tag}.{field}.rel_dev",
                        deviation,
                        tensor_config.output_rtol,
                    )
                )
        if tensor_config.expected_backend is not None:
            observed = case.get("direct_backend")
            match = (
                0.0
                if isinstance(observed, str) and observed == tensor_config.expected_backend
                else None
            )
            checks.append(
                _check(
                    f"tensor_cost.{tag}.backend",
                    match,
                    0.0,
                    detail=f"expected {tensor_config.expected_backend!r}, got {observed!r}",
                )
            )

    checks.append(
        _check(
            "flagship.median_s",
            flagship_artifact.median_s,
            flagship_config.reference_median_s * flagship_config.absolute_multiple,
        )
    )
    expected = flagship_config.expected
    if expected is not None:
        rtol = expected["rtol"]
        assert isinstance(rtol, float)
        for field in ("deviance", "effective_df"):
            measured_value = _finite_real(flagship_artifact.payload.get(field))
            expected_value = expected[field]
            assert isinstance(expected_value, float)
            deviation = (
                None if measured_value is None else abs(measured_value / expected_value - 1.0)
            )
            checks.append(_check(f"flagship.{field}.rel_dev", deviation, rtol))
        backend_expected = expected.get("backend")
        if isinstance(backend_expected, str):
            observed = flagship_artifact.payload.get("direct_backend")
            match = 0.0 if isinstance(observed, str) and observed == backend_expected else None
            checks.append(
                _check(
                    "flagship.backend",
                    match,
                    0.0,
                    detail=f"expected {backend_expected!r}, got {observed!r}",
                )
            )
    if flagship_config.expected_n is not None:
        measured_n = _exact_int(flagship_artifact.payload.get("n"), positive=True)
        checks.append(
            _check(
                "flagship.n",
                None if measured_n is None else abs(measured_n - flagship_config.expected_n),
                0.0,
            )
        )
    return checks


def evaluate_gate(
    baselines: object,
    tensor: object,
    flagship: object,
    *,
    machine_profile: str | None,
) -> list[GateCheck]:
    """Evaluate a certified profile without throwing on malformed JSON data."""
    certification = certification_check(baselines, machine_profile)
    if not certification.passed:
        return [certification]
    try:
        return _evaluate_certified(baselines, tensor, flagship, certification)
    except Exception as exc:  # noqa: BLE001 - fail closed on adversarial JSON shapes
        return [
            certification,
            _failure(
                "gate.validation",
                f"artifact validation raised {type(exc).__name__}: {exc}",
            ),
        ]


def _ci_runtime_variable(environ: Mapping[str, str] | None = None) -> str | None:
    environment = os.environ if environ is None else environ
    for name in ("GITHUB_ACTIONS", "CI"):
        value = environment.get(name)
        if isinstance(value, str) and value.strip().lower() in _TRUTHY_ENV_VALUES:
            return name
    return None


def _load_json(path: str, label: str) -> tuple[object | None, str | None]:
    try:
        return json.loads(Path(path).read_text()), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"cannot load {label} {path!r}: {type(exc).__name__}: {exc}"


def _render_result(checks: Sequence[GateCheck], baselines_path: str) -> int:
    print("Local performance gate:")
    for check in checks:
        print(check.render())
    failed = [check for check in checks if not check.passed]
    if failed:
        print(f"\n{len(failed)} check(s) FAILED against {baselines_path}")
        return 1
    print(f"\nAll {len(checks)} checks passed against {baselines_path}")
    return 0


def main() -> int:
    ci_variable = _ci_runtime_variable()
    if ci_variable is not None:
        print(
            f"REFUSED: local wall-time certification cannot run when {ci_variable} is truthy",
            file=sys.stderr,
        )
        return 2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--machine-profile",
        required=True,
        help="operator-asserted local profile ID; must match an enabled baseline",
    )
    parser.add_argument("--baselines", required=True)
    parser.add_argument("--tensor-json", required=True)
    parser.add_argument("--flagship-json", required=True)
    args = parser.parse_args()

    baselines, error = _load_json(args.baselines, "baseline")
    if error is not None:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 1
    certification = certification_check(baselines, args.machine_profile)
    if not certification.passed:
        return _render_result([certification], args.baselines)

    tensor, error = _load_json(args.tensor_json, "tensor artifact")
    if error is not None:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 1
    flagship, error = _load_json(args.flagship_json, "flagship artifact")
    if error is not None:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 1

    checks = evaluate_gate(
        baselines,
        tensor=tensor,
        flagship=flagship,
        machine_profile=args.machine_profile,
    )
    return _render_result(checks, args.baselines)


if __name__ == "__main__":
    sys.exit(main())
