"""Validate neutral Tweedie reference fixtures and JSON-lines candidates.

The wire protocol sends one versioned density request per line and requires one
versioned response with the same case identifier.  Candidate commands receive
no expected values.  Both checker modes also replay and validate every profile
case locally.  Profile recipe ``normal_log_mean.v1`` means that one
``default_rng(seed)`` instance draws ``x = standard_normal(n)``, constructs
``mu = exp(log_mu_intercept + log_mu_slope * x)``, and then draws the response
from the published compound-Poisson/Gamma recipe using that same generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "tweedie_reference_values.json"
REFERENCE_FORMAT = "superglm.tweedie.reference.v1"
REQUEST_FORMAT = "superglm.tweedie.reference.request.v1"
RESPONSE_FORMAT = "superglm.tweedie.reference.response.v1"
_CASE_ID = re.compile(r"[a-z0-9][a-z0-9_-]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ReferenceCheckError(RuntimeError):
    """Base class for an expected checker failure."""

    exit_code = 2


class ReferenceSchemaError(ReferenceCheckError):
    """The fixture or checker configuration is invalid."""


class ReferenceComparisonError(ReferenceCheckError):
    """Valid numeric results exceed the committed tolerance."""

    exit_code = 1


class ReferenceExecutionError(ReferenceCheckError):
    """The external reference command could not complete successfully."""

    exit_code = 3


class ReferenceProtocolError(ReferenceCheckError):
    """The external command violated the JSON-lines protocol."""

    exit_code = 4


@dataclass(frozen=True, slots=True)
class ReferenceTolerances:
    logpdf_abs: float
    p_abs: float
    phi_rel: float


@dataclass(frozen=True, slots=True)
class DensityReferenceCase:
    case: str
    y: float
    mu: float
    phi: float
    p: float
    weight: float
    logpdf: float


@dataclass(frozen=True, slots=True)
class ProfileReferenceCase:
    case: str
    seed: int
    n: int
    recipe: str
    log_mu_intercept: float
    log_mu_slope: float
    true_p: float
    true_phi: float
    reference_p: float
    reference_phi: float
    reference_converged: bool
    response_sha256: str


@dataclass(frozen=True, slots=True)
class ReferenceFixture:
    format: str
    tolerances: ReferenceTolerances
    density_cases: tuple[DensityReferenceCase, ...]
    profile_cases: tuple[ProfileReferenceCase, ...]


@dataclass(frozen=True, slots=True)
class ComparisonSummary:
    n_cases: int
    tolerance: float
    max_local_fixture_abs: float
    worst_local_fixture_case: str
    max_candidate_fixture_abs: float | None = None
    worst_candidate_fixture_case: str | None = None
    max_candidate_local_abs: float | None = None
    worst_candidate_local_case: str | None = None
    n_profile_cases: int = 0
    p_tolerance: float | None = None
    max_local_profile_p_abs: float | None = None
    worst_local_profile_p_case: str | None = None
    phi_rel_tolerance: float | None = None
    max_local_profile_phi_rel: float | None = None
    worst_local_profile_phi_case: str | None = None

    def render(self) -> str:
        fields = [
            f"checked={self.n_cases}",
            f"tolerance={self.tolerance:.17g}",
            f"max_local_fixture_abs={self.max_local_fixture_abs:.17g}",
            f"worst_local_fixture_case={self.worst_local_fixture_case}",
        ]
        if self.max_candidate_fixture_abs is not None:
            fields.extend(
                [
                    f"max_candidate_fixture_abs={self.max_candidate_fixture_abs:.17g}",
                    f"worst_candidate_fixture_case={self.worst_candidate_fixture_case}",
                    f"max_candidate_local_abs={self.max_candidate_local_abs:.17g}",
                    f"worst_candidate_local_case={self.worst_candidate_local_case}",
                ]
            )
        if self.n_profile_cases:
            fields.extend(
                [
                    f"profile_checked={self.n_profile_cases}",
                    f"p_tolerance={self.p_tolerance:.17g}",
                    f"max_local_profile_p_abs={self.max_local_profile_p_abs:.17g}",
                    f"worst_local_profile_p_case={self.worst_local_profile_p_case}",
                    f"phi_rel_tolerance={self.phi_rel_tolerance:.17g}",
                    f"max_local_profile_phi_rel={self.max_local_profile_phi_rel:.17g}",
                    f"worst_local_profile_phi_case={self.worst_local_profile_phi_case}",
                    "profile_response_digests=verified",
                    "profile_convergence=verified",
                ]
            )
        return " ".join(fields)


@dataclass(frozen=True, slots=True)
class _ProfileComparisonSummary:
    n_cases: int
    max_p_abs: float
    worst_p_case: str
    max_phi_rel: float
    worst_phi_case: str


def _strict_json_loads(
    text: str,
    *,
    label: str,
    error_type: type[ReferenceCheckError],
) -> object:
    def reject_constant(value: str) -> None:
        raise error_type(f"{label}: non-finite JSON constant {value!r} is not allowed")

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise error_type(f"{label}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except error_type:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise error_type(f"{label}: invalid JSON: {exc}") from exc


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ReferenceSchemaError(f"{label} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if any(not isinstance(key, str) for key in value):
        raise ReferenceSchemaError(f"{label} keys must all be JSON strings")
    keys = set(value)
    missing = sorted(expected - keys)
    unexpected = sorted(keys - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise ReferenceSchemaError(f"{label} has invalid keys: {', '.join(details)}")


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or type(value) not in {int, float}:
        raise ReferenceSchemaError(f"{label} must be one finite JSON number, excluding booleans")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ReferenceSchemaError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ReferenceSchemaError(f"{label} must be finite")
    return result


def _positive_number(value: object, label: str) -> float:
    result = _finite_number(value, label)
    if result <= 0.0:
        raise ReferenceSchemaError(f"{label} must be strictly positive")
    return result


def _power(value: object, label: str) -> float:
    result = _finite_number(value, label)
    if not 1.0 < result < 2.0:
        raise ReferenceSchemaError(f"{label} must be in the open interval (1, 2)")
    return result


def _integer(value: object, label: str, *, minimum: int) -> int:
    if type(value) is not int:
        raise ReferenceSchemaError(f"{label} must be one JSON integer, excluding booleans")
    if value < minimum:
        raise ReferenceSchemaError(f"{label} must be at least {minimum}")
    return value


def _case_id(value: object, label: str) -> str:
    if not isinstance(value, str) or _CASE_ID.fullmatch(value) is None:
        raise ReferenceSchemaError(f"{label} must match '[a-z0-9][a-z0-9_-]*'")
    return value


def _parse_tolerances(value: object) -> ReferenceTolerances:
    mapping = _mapping(value, "tolerances")
    _exact_keys(mapping, {"logpdf_abs", "p_abs", "phi_rel"}, "tolerances")
    logpdf_abs = _positive_number(mapping["logpdf_abs"], "tolerances.logpdf_abs")
    p_abs = _positive_number(mapping["p_abs"], "tolerances.p_abs")
    phi_rel = _positive_number(mapping["phi_rel"], "tolerances.phi_rel")
    caps = {
        "logpdf_abs": (logpdf_abs, 1e-8),
        "p_abs": (p_abs, 2e-4),
        "phi_rel": (phi_rel, 5e-4),
    }
    for name, (actual, cap) in caps.items():
        if actual > cap:
            raise ReferenceSchemaError(
                f"tolerances.{name}={actual!r} exceeds the format-v1 cap {cap!r}"
            )
    return ReferenceTolerances(logpdf_abs=logpdf_abs, p_abs=p_abs, phi_rel=phi_rel)


def _parse_density_case(value: object, index: int) -> DensityReferenceCase:
    label = f"density_cases[{index}]"
    mapping = _mapping(value, label)
    _exact_keys(mapping, {"case", "y", "mu", "phi", "p", "weight", "logpdf"}, label)
    y = _finite_number(mapping["y"], f"{label}.y")
    if y < 0.0:
        raise ReferenceSchemaError(f"{label}.y must be nonnegative")
    return DensityReferenceCase(
        case=_case_id(mapping["case"], f"{label}.case"),
        y=y,
        mu=_positive_number(mapping["mu"], f"{label}.mu"),
        phi=_positive_number(mapping["phi"], f"{label}.phi"),
        p=_power(mapping["p"], f"{label}.p"),
        weight=_positive_number(mapping["weight"], f"{label}.weight"),
        logpdf=_finite_number(mapping["logpdf"], f"{label}.logpdf"),
    )


def _parse_profile_case(value: object, index: int) -> ProfileReferenceCase:
    label = f"profile_cases[{index}]"
    mapping = _mapping(value, label)
    expected = {
        "case",
        "seed",
        "n",
        "recipe",
        "log_mu_intercept",
        "log_mu_slope",
        "true_p",
        "true_phi",
        "reference_p",
        "reference_phi",
        "reference_converged",
        "response_sha256",
    }
    _exact_keys(mapping, expected, label)
    if mapping["recipe"] != "normal_log_mean.v1":
        raise ReferenceSchemaError(f"{label}.recipe must be 'normal_log_mean.v1'")
    if type(mapping["reference_converged"]) is not bool:
        raise ReferenceSchemaError(f"{label}.reference_converged must be a JSON boolean")
    response_sha256 = mapping["response_sha256"]
    if not isinstance(response_sha256, str) or _SHA256.fullmatch(response_sha256) is None:
        raise ReferenceSchemaError(f"{label}.response_sha256 must be 64 lowercase hex digits")
    return ProfileReferenceCase(
        case=_case_id(mapping["case"], f"{label}.case"),
        seed=_integer(mapping["seed"], f"{label}.seed", minimum=0),
        n=_integer(mapping["n"], f"{label}.n", minimum=1),
        recipe="normal_log_mean.v1",
        log_mu_intercept=_finite_number(mapping["log_mu_intercept"], f"{label}.log_mu_intercept"),
        log_mu_slope=_finite_number(mapping["log_mu_slope"], f"{label}.log_mu_slope"),
        true_p=_power(mapping["true_p"], f"{label}.true_p"),
        true_phi=_positive_number(mapping["true_phi"], f"{label}.true_phi"),
        reference_p=_power(mapping["reference_p"], f"{label}.reference_p"),
        reference_phi=_positive_number(mapping["reference_phi"], f"{label}.reference_phi"),
        reference_converged=bool(mapping["reference_converged"]),
        response_sha256=response_sha256,
    )


def validate_reference_payload(payload: object) -> ReferenceFixture:
    mapping = _mapping(payload, "fixture")
    _exact_keys(
        mapping,
        {"format", "tolerances", "density_cases", "profile_cases"},
        "fixture",
    )
    if mapping["format"] != REFERENCE_FORMAT:
        raise ReferenceSchemaError(f"fixture.format must be {REFERENCE_FORMAT!r}")
    density_values = mapping["density_cases"]
    profile_values = mapping["profile_cases"]
    if not isinstance(density_values, list) or not density_values:
        raise ReferenceSchemaError("density_cases must be one nonempty JSON array")
    if not isinstance(profile_values, list) or not profile_values:
        raise ReferenceSchemaError("profile_cases must be one nonempty JSON array")
    density_cases = tuple(
        _parse_density_case(value, index) for index, value in enumerate(density_values)
    )
    profile_cases = tuple(
        _parse_profile_case(value, index) for index, value in enumerate(profile_values)
    )
    case_ids = [case.case for case in density_cases] + [case.case for case in profile_cases]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for case in case_ids:
        if case in seen:
            duplicates.add(case)
        seen.add(case)
    if duplicates:
        raise ReferenceSchemaError(f"fixture case identifiers must be unique: {sorted(duplicates)}")
    return ReferenceFixture(
        format=REFERENCE_FORMAT,
        tolerances=_parse_tolerances(mapping["tolerances"]),
        density_cases=density_cases,
        profile_cases=profile_cases,
    )


def load_reference_fixture(path: Path) -> ReferenceFixture:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ReferenceSchemaError(f"could not read fixture {path}: {exc}") from exc
    payload = _strict_json_loads(
        text,
        label=str(path),
        error_type=ReferenceSchemaError,
    )
    return validate_reference_payload(payload)


def _local_logpdf(fixture: ReferenceFixture) -> tuple[float, ...]:
    from superglm._tweedie_density import TweedieDensityError
    from superglm.profiling.tweedie import tweedie_logpdf

    values = []
    for case in fixture.density_cases:
        try:
            result = tweedie_logpdf(
                np.array([case.y]),
                np.array([case.mu]),
                case.phi,
                case.p,
                weights=np.array([case.weight]),
            )
        except TweedieDensityError as exc:
            raise ReferenceComparisonError(
                f"local exact density case={case.case} could not certify: {exc}"
            ) from exc
        value = float(result[0])
        if not math.isfinite(value):
            raise ReferenceComparisonError(
                f"local exact density returned non-finite output for {case.case}"
            )
        values.append(value)
    return tuple(values)


def _regenerate_profile_data(case: ProfileReferenceCase) -> tuple[np.ndarray, np.ndarray]:
    from superglm.profiling.tweedie import generate_tweedie_cpg

    try:
        rng = np.random.default_rng(case.seed)
        x = rng.standard_normal(case.n)
        with np.errstate(over="ignore", invalid="ignore"):
            mu = np.exp(case.log_mu_intercept + case.log_mu_slope * x)
        if not np.all(np.isfinite(mu)) or np.any(mu <= 0.0):
            raise ValueError("generated means must be finite and strictly positive")
        y = generate_tweedie_cpg(
            case.n,
            mu=mu,
            phi=case.true_phi,
            p=case.true_p,
            rng=rng,
        )
    except (TypeError, ValueError, OverflowError, FloatingPointError, RuntimeError) as exc:
        raise ReferenceComparisonError(
            f"local profile replay case={case.case} failed: {exc}"
        ) from exc
    return x, y


def _response_sha256(y: np.ndarray) -> str:
    canonical = np.ascontiguousarray(y, dtype=np.dtype("<f8"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _fit_local_profile(
    case: ProfileReferenceCase,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, bool]:
    import pandas as pd

    from superglm import SuperGLM
    from superglm.distributions import Tweedie
    from superglm.features.numeric import Numeric
    from superglm.profiling.tweedie import estimate_tweedie_p

    model = SuperGLM(
        family=Tweedie(1.5),
        selection_penalty=0.0,
        features={"x": Numeric()},
    )
    try:
        result = estimate_tweedie_p(
            model,
            pd.DataFrame({"x": x}),
            y,
            phi_method="mle",
            method="brent",
        )
        p_hat = float(result.p_hat)
        phi_hat = float(result.phi_hat)
    except (TypeError, ValueError, OverflowError, FloatingPointError, RuntimeError) as exc:
        raise ReferenceComparisonError(f"local profile fit case={case.case} failed: {exc}") from exc
    if not math.isfinite(p_hat) or not math.isfinite(phi_hat) or phi_hat <= 0.0:
        raise ReferenceComparisonError(
            f"local profile fit case={case.case} returned invalid p or phi"
        )
    return p_hat, phi_hat, bool(result.converged)


def _check_local_profiles(fixture: ReferenceFixture) -> _ProfileComparisonSummary:
    observed_p = []
    observed_phi = []
    observed_converged = []
    for case in fixture.profile_cases:
        x, y = _regenerate_profile_data(case)
        digest = _response_sha256(y)
        if digest != case.response_sha256:
            raise ReferenceComparisonError(
                f"profile response digest case={case.case} observed={digest!r} "
                f"expected={case.response_sha256!r}"
            )
        p_hat, phi_hat, converged = _fit_local_profile(case, x, y)
        observed_p.append(p_hat)
        observed_phi.append(phi_hat)
        observed_converged.append(converged)

    p_errors = [
        abs(actual - case.reference_p)
        for case, actual in zip(fixture.profile_cases, observed_p, strict=True)
    ]
    phi_errors = [
        abs(actual - case.reference_phi) / case.reference_phi
        for case, actual in zip(fixture.profile_cases, observed_phi, strict=True)
    ]
    failures = []
    for case, actual, error in zip(fixture.profile_cases, observed_p, p_errors, strict=True):
        if error > fixture.tolerances.p_abs:
            failures.append(
                f"profile-p local-vs-fixture case={case.case} observed={actual:.17g} "
                f"expected={case.reference_p:.17g} abs_error={error:.17g} "
                f"tolerance={fixture.tolerances.p_abs:.17g}"
            )
    for case, actual, error in zip(
        fixture.profile_cases,
        observed_phi,
        phi_errors,
        strict=True,
    ):
        if error > fixture.tolerances.phi_rel:
            failures.append(
                f"profile-phi local-vs-fixture case={case.case} observed={actual:.17g} "
                f"expected={case.reference_phi:.17g} rel_error={error:.17g} "
                f"tolerance={fixture.tolerances.phi_rel:.17g}"
            )
    for case, actual in zip(fixture.profile_cases, observed_converged, strict=True):
        if actual is not case.reference_converged:
            failures.append(
                f"profile-converged local-vs-fixture case={case.case} "
                f"observed={actual!r} expected={case.reference_converged!r}"
            )
    if failures:
        raise ReferenceComparisonError("\n".join(failures))

    worst_p_index = max(range(len(p_errors)), key=p_errors.__getitem__)
    worst_phi_index = max(range(len(phi_errors)), key=phi_errors.__getitem__)
    return _ProfileComparisonSummary(
        n_cases=len(fixture.profile_cases),
        max_p_abs=p_errors[worst_p_index],
        worst_p_case=fixture.profile_cases[worst_p_index].case,
        max_phi_rel=phi_errors[worst_phi_index],
        worst_phi_case=fixture.profile_cases[worst_phi_index].case,
    )


def _max_error(
    cases: tuple[DensityReferenceCase, ...],
    observed: Sequence[float],
    expected: Sequence[float],
) -> tuple[float, str]:
    errors = [abs(left - right) for left, right in zip(observed, expected, strict=True)]
    worst_index = max(range(len(errors)), key=errors.__getitem__)
    return errors[worst_index], cases[worst_index].case


def _require_within_tolerance(
    *,
    metric: str,
    cases: tuple[DensityReferenceCase, ...],
    observed: Sequence[float],
    expected: Sequence[float],
    tolerance: float,
) -> tuple[float, str]:
    maximum, worst_case = _max_error(cases, observed, expected)
    if maximum > tolerance:
        failures = []
        for case, actual, target in zip(cases, observed, expected, strict=True):
            error = abs(actual - target)
            if error > tolerance:
                failures.append(
                    f"{metric} case={case.case} observed={actual:.17g} "
                    f"expected={target:.17g} abs_error={error:.17g} "
                    f"tolerance={tolerance:.17g}"
                )
        raise ReferenceComparisonError("\n".join(failures))
    return maximum, worst_case


def run_self_check(fixture: ReferenceFixture) -> ComparisonSummary:
    local = _local_logpdf(fixture)
    committed = tuple(case.logpdf for case in fixture.density_cases)
    maximum, worst_case = _require_within_tolerance(
        metric="local-vs-fixture",
        cases=fixture.density_cases,
        observed=local,
        expected=committed,
        tolerance=fixture.tolerances.logpdf_abs,
    )
    profiles = _check_local_profiles(fixture)
    return ComparisonSummary(
        n_cases=len(fixture.density_cases),
        tolerance=fixture.tolerances.logpdf_abs,
        max_local_fixture_abs=maximum,
        worst_local_fixture_case=worst_case,
        n_profile_cases=profiles.n_cases,
        p_tolerance=fixture.tolerances.p_abs,
        max_local_profile_p_abs=profiles.max_p_abs,
        worst_local_profile_p_case=profiles.worst_p_case,
        phi_rel_tolerance=fixture.tolerances.phi_rel,
        max_local_profile_phi_rel=profiles.max_phi_rel,
        worst_local_profile_phi_case=profiles.worst_phi_case,
    )


def _request_payload(fixture: ReferenceFixture) -> str:
    lines = []
    for case in fixture.density_cases:
        request = {
            "format": REQUEST_FORMAT,
            "case": case.case,
            "y": case.y,
            "mu": case.mu,
            "phi": case.phi,
            "p": case.p,
            "weight": case.weight,
        }
        lines.append(json.dumps(request, separators=(",", ":"), allow_nan=False))
    return "\n".join(lines) + "\n"


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        elif process.poll() is None:
            process.kill()
    except ProcessLookupError:
        pass


def _kill_and_reap(process: subprocess.Popen[str]) -> None:
    _kill_process_group(process)
    try:
        process.communicate(timeout=5.0)
    except (subprocess.TimeoutExpired, UnicodeError):
        process.kill()
        process.wait()


def _run_external_command(argv: Sequence[str], request: str, timeout_s: float) -> str:
    if (
        isinstance(argv, str | bytes)
        or not argv
        or not isinstance(argv[0], str)
        or not argv[0]
        or any(not isinstance(item, str) for item in argv[1:])
    ):
        raise ReferenceSchemaError("--command requires a nonempty executable and string arguments")
    try:
        process = subprocess.Popen(
            list(argv),
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            start_new_session=os.name == "posix",
        )
    except OSError as exc:
        raise ReferenceExecutionError(f"could not launch reference command: {exc}") from exc
    try:
        stdout, stderr = process.communicate(request, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        _kill_and_reap(process)
        raise ReferenceExecutionError(
            f"reference command exceeded timeout {timeout_s:.17g}s"
        ) from exc
    except UnicodeError as exc:
        _kill_and_reap(process)
        raise ReferenceProtocolError("reference command output was not valid UTF-8") from exc
    except OSError as exc:
        _kill_and_reap(process)
        raise ReferenceExecutionError(f"reference command communication failed: {exc}") from exc
    _kill_process_group(process)
    if process.returncode != 0:
        stderr_excerpt = stderr.strip().replace("\n", " ")[:500]
        raise ReferenceExecutionError(
            f"reference command exited with status {process.returncode}; stderr={stderr_excerpt!r}"
        )
    return stdout


def _parse_responses(fixture: ReferenceFixture, stdout: str) -> tuple[float, ...]:
    lines = stdout.splitlines()
    if len(lines) != len(fixture.density_cases):
        raise ReferenceProtocolError(
            "reference command returned "
            f"{len(lines)} lines for {len(fixture.density_cases)} requests"
        )
    values = []
    expected_keys = {"format", "case", "logpdf"}
    for index, (line, case) in enumerate(zip(lines, fixture.density_cases, strict=True)):
        label = f"response line {index + 1}"
        payload = _strict_json_loads(
            line,
            label=label,
            error_type=ReferenceProtocolError,
        )
        if not isinstance(payload, Mapping):
            raise ReferenceProtocolError(f"{label} must be a JSON object")
        keys = set(payload)
        if keys != expected_keys:
            raise ReferenceProtocolError(
                f"{label} has invalid keys; expected={sorted(expected_keys)}, actual={sorted(keys)}"
            )
        if payload["format"] != RESPONSE_FORMAT:
            raise ReferenceProtocolError(f"{label} has an invalid response format")
        if payload["case"] != case.case:
            raise ReferenceProtocolError(
                f"{label} case mismatch: expected={case.case!r}, actual={payload['case']!r}"
            )
        value = payload["logpdf"]
        if isinstance(value, bool) or type(value) not in {int, float}:
            raise ReferenceProtocolError(f"{label}.logpdf must be one finite JSON number")
        try:
            numeric = float(value)
        except (OverflowError, ValueError) as exc:
            raise ReferenceProtocolError(f"{label}.logpdf must be finite") from exc
        if not math.isfinite(numeric):
            raise ReferenceProtocolError(f"{label}.logpdf must be finite")
        values.append(numeric)
    return tuple(values)


def run_command(
    fixture: ReferenceFixture,
    argv: Sequence[str],
    *,
    timeout_s: float = 30.0,
) -> ComparisonSummary:
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, int | float):
        raise ReferenceSchemaError("timeout must be one finite positive number")
    try:
        timeout = float(timeout_s)
    except (OverflowError, ValueError) as exc:
        raise ReferenceSchemaError("timeout must be one finite positive number") from exc
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ReferenceSchemaError("timeout must be one finite positive number")

    local = _local_logpdf(fixture)
    committed = tuple(case.logpdf for case in fixture.density_cases)
    local_maximum, local_worst = _require_within_tolerance(
        metric="local-vs-fixture",
        cases=fixture.density_cases,
        observed=local,
        expected=committed,
        tolerance=fixture.tolerances.logpdf_abs,
    )
    profiles = _check_local_profiles(fixture)
    stdout = _run_external_command(argv, _request_payload(fixture), timeout)
    candidate = _parse_responses(fixture, stdout)
    candidate_fixture_maximum, candidate_fixture_worst = _require_within_tolerance(
        metric="candidate-vs-fixture",
        cases=fixture.density_cases,
        observed=candidate,
        expected=committed,
        tolerance=fixture.tolerances.logpdf_abs,
    )
    candidate_local_maximum, candidate_local_worst = _require_within_tolerance(
        metric="candidate-vs-local",
        cases=fixture.density_cases,
        observed=candidate,
        expected=local,
        tolerance=fixture.tolerances.logpdf_abs,
    )
    return ComparisonSummary(
        n_cases=len(fixture.density_cases),
        tolerance=fixture.tolerances.logpdf_abs,
        max_local_fixture_abs=local_maximum,
        worst_local_fixture_case=local_worst,
        max_candidate_fixture_abs=candidate_fixture_maximum,
        worst_candidate_fixture_case=candidate_fixture_worst,
        max_candidate_local_abs=candidate_local_maximum,
        worst_candidate_local_case=candidate_local_worst,
        n_profile_cases=profiles.n_cases,
        p_tolerance=fixture.tolerances.p_abs,
        max_local_profile_p_abs=profiles.max_p_abs,
        worst_local_profile_p_case=profiles.worst_p_case,
        phi_rel_tolerance=fixture.tolerances.phi_rel,
        max_local_profile_phi_rel=profiles.max_phi_rel,
        worst_local_profile_phi_case=profiles.worst_phi_case,
    )


def _positive_timeout(value: str) -> float:
    try:
        timeout = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timeout must be finite and positive") from exc
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise argparse.ArgumentTypeError("timeout must be finite and positive")
    return timeout


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check neutral Tweedie density and profile reference records.",
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--timeout", type=_positive_timeout, default=30.0)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-check", action="store_true")
    mode.add_argument(
        "--command",
        nargs=argparse.REMAINDER,
        help="external JSON-lines command argv; this option must be last",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        fixture = load_reference_fixture(args.fixture)
        if args.self_check:
            summary = run_self_check(fixture)
        else:
            summary = run_command(fixture, args.command or (), timeout_s=args.timeout)
    except ReferenceCheckError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code
    print(summary.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
