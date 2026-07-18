from __future__ import annotations

import copy
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scripts.check_tweedie_reference import (
    RESPONSE_FORMAT,
    ReferenceComparisonError,
    ReferenceExecutionError,
    ReferenceProtocolError,
    ReferenceSchemaError,
    load_reference_fixture,
    run_command,
    run_self_check,
    validate_reference_payload,
)

from superglm import SuperGLM
from superglm._tweedie_density import evaluate_tweedie_density
from superglm.distributions import Tweedie
from superglm.features.numeric import Numeric
from superglm.profiling.tweedie import estimate_tweedie_p, generate_tweedie_cpg, tweedie_logpdf

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "tweedie_reference_values.json"
CHECKER_PATH = ROOT / "scripts" / "check_tweedie_reference.py"
COMMITTED_FIXTURE = load_reference_fixture(FIXTURE_PATH)


def _regenerate_profile_response(profile):
    rng = np.random.default_rng(profile.seed)
    x = rng.standard_normal(profile.n)
    mu = np.exp(profile.log_mu_intercept + profile.log_mu_slope * x)
    y = generate_tweedie_cpg(
        profile.n,
        mu=mu,
        phi=profile.true_phi,
        p=profile.true_p,
        rng=rng,
    )
    return x, y


def _response_sha256(y: np.ndarray) -> str:
    canonical = np.ascontiguousarray(y, dtype=np.dtype("<f8"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _valid_payload() -> dict[str, object]:
    return {
        "format": "superglm.tweedie.reference.v1",
        "tolerances": {
            "logpdf_abs": 1e-8,
            "p_abs": 2e-4,
            "phi_rel": 5e-4,
        },
        "density_cases": [
            {
                "case": "zero_atom",
                "y": 0.0,
                "mu": 1.0,
                "phi": 1.0,
                "p": 1.5,
                "weight": 1.0,
                "logpdf": -2.0,
            }
        ],
        "profile_cases": [
            {
                "case": "seed_101_low_power",
                "seed": 101,
                "n": 800,
                "recipe": "normal_log_mean.v1",
                "log_mu_intercept": 0.3,
                "log_mu_slope": 0.45,
                "true_p": 1.2,
                "true_phi": 0.8,
                "reference_p": 1.196897,
                "reference_phi": 0.81,
                "reference_converged": True,
                "response_sha256": "0" * 64,
            }
        ],
    }


def test_reference_payload_is_strictly_validated_and_detached() -> None:
    payload = _valid_payload()
    fixture = validate_reference_payload(payload)

    assert fixture.format == "superglm.tweedie.reference.v1"
    assert isinstance(fixture.density_cases, tuple)
    assert isinstance(fixture.profile_cases, tuple)
    assert fixture.density_cases[0].case == "zero_atom"
    assert fixture.profile_cases[0].reference_phi == 0.81

    payload["density_cases"][0]["logpdf"] = 100.0  # type: ignore[index]
    assert fixture.density_cases[0].logpdf == -2.0
    with pytest.raises(FrozenInstanceError):
        fixture.tolerances.logpdf_abs = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda payload: payload.update(extra=True), "unexpected"),
        (
            lambda payload: payload["density_cases"][0].update(phi=True),  # type: ignore[index]
            "phi",
        ),
        (
            lambda payload: payload["density_cases"][0].update(logpdf=float("inf")),  # type: ignore[index]
            "logpdf",
        ),
        (
            lambda payload: payload["density_cases"][0].update(phi=10**400),  # type: ignore[index]
            "phi",
        ),
        (
            lambda payload: payload["profile_cases"][0].update(seed=True),  # type: ignore[index]
            "seed",
        ),
        (
            lambda payload: payload["profile_cases"][0].update(case="zero_atom"),  # type: ignore[index]
            "unique",
        ),
        (
            lambda payload: payload["tolerances"].update(logpdf_abs=1e-7),  # type: ignore[union-attr]
            "logpdf_abs",
        ),
    ],
)
def test_reference_payload_rejects_unsafe_schema_mutations(mutation, match) -> None:
    payload = copy.deepcopy(_valid_payload())
    mutation(payload)

    with pytest.raises(ReferenceSchemaError, match=match):
        validate_reference_payload(payload)


@pytest.mark.parametrize(
    "raw",
    [
        '{"format":"superglm.tweedie.reference.v1","format":"duplicate"}',
        '{"format":"superglm.tweedie.reference.v1","value":NaN}',
        '{"format":"superglm.tweedie.reference.v1","value":Infinity}',
    ],
)
def test_fixture_loader_rejects_nonstandard_or_ambiguous_json(tmp_path, raw) -> None:
    path = tmp_path / "fixture.json"
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(ReferenceSchemaError):
        load_reference_fixture(path)


def test_self_check_uses_strict_absolute_error() -> None:
    fixture = validate_reference_payload(_valid_payload())

    summary = run_self_check(fixture)

    assert summary.n_cases == 1
    assert summary.max_local_fixture_abs == 0.0
    assert summary.max_candidate_fixture_abs is None


def test_self_check_reports_uncertifiable_local_case_without_raw_kernel_error() -> None:
    payload = _valid_payload()
    payload["density_cases"][0].update(  # type: ignore[index]
        y=1.0,
        mu=1.0,
        phi=float(np.nextafter(0.0, 1.0)),
        p=1.6,
        weight=1.0,
        logpdf=0.0,
    )
    fixture = validate_reference_payload(payload)

    with pytest.raises(ReferenceComparisonError, match="zero_atom.*could not certify"):
        run_self_check(fixture)


def test_json_lines_command_receives_versioned_input_and_echoes_case() -> None:
    fixture = validate_reference_payload(_valid_payload())
    code = f"""
import json
import sys

assert sys.argv[1] == "argv token with spaces"
assert sys.argv[2] == ""
for line in sys.stdin:
    request = json.loads(line)
    assert request["format"] == "superglm.tweedie.reference.request.v1"
    response = {{
        "format": {RESPONSE_FORMAT!r},
        "case": request["case"],
        "logpdf": -2.0,
    }}
    print(json.dumps(response, separators=(",", ":")))
"""

    summary = run_command(
        fixture,
        [sys.executable, "-c", code, "argv token with spaces", ""],
    )

    assert summary.max_candidate_fixture_abs == 0.0
    assert summary.max_candidate_local_abs == 0.0


@pytest.mark.parametrize(
    ("code", "match"),
    [
        (
            f"""
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps({{"format": {RESPONSE_FORMAT!r}, "case": request["case"], "logpdf": True}}))
""",
            "logpdf",
        ),
        (
            f"""
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps({{"format": {RESPONSE_FORMAT!r}, "case": "wrong", "logpdf": -2.0}}))
""",
            "case mismatch",
        ),
        (
            f"""
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    print("unexpected log line")
    print(json.dumps({{"format": {RESPONSE_FORMAT!r}, "case": request["case"], "logpdf": -2.0}}))
""",
            "returned 2 lines",
        ),
        (
            f"""
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    print('{{"format":"{RESPONSE_FORMAT}","case":"zero_atom","logpdf":-2.0,"logpdf":-2.0}}')
""",
            "duplicate JSON key",
        ),
    ],
)
def test_json_lines_command_rejects_malformed_protocol(code, match) -> None:
    fixture = validate_reference_payload(_valid_payload())

    with pytest.raises(ReferenceProtocolError, match=match):
        run_command(fixture, [sys.executable, "-c", code])


def test_json_lines_command_rejects_valid_but_wrong_numeric_output() -> None:
    fixture = validate_reference_payload(_valid_payload())
    code = f"""
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps({{"format": {RESPONSE_FORMAT!r}, "case": request["case"], "logpdf": -1.0}}))
"""

    with pytest.raises(ReferenceComparisonError, match="candidate-vs-fixture"):
        run_command(fixture, [sys.executable, "-c", code])


def test_json_lines_command_rejects_integer_too_large_for_float64() -> None:
    fixture = validate_reference_payload(_valid_payload())
    code = f"""
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps({{"format": {RESPONSE_FORMAT!r}, "case": request["case"], "logpdf": 10**400}}))
"""

    with pytest.raises(ReferenceProtocolError, match="logpdf"):
        run_command(fixture, [sys.executable, "-c", code])


def test_json_lines_command_rejects_timeout_too_large_for_float64() -> None:
    fixture = validate_reference_payload(_valid_payload())

    with pytest.raises(ReferenceSchemaError, match="timeout"):
        run_command(fixture, [sys.executable, "-c", "pass"], timeout_s=10**400)


def test_json_lines_command_reports_nonzero_exit_and_stderr() -> None:
    fixture = validate_reference_payload(_valid_payload())
    code = "import sys; print('neutral failure', file=sys.stderr); raise SystemExit(7)"

    with pytest.raises(ReferenceExecutionError, match="status 7.*neutral failure"):
        run_command(fixture, [sys.executable, "-c", code])


def test_json_lines_command_kills_a_timed_out_process_group() -> None:
    fixture = validate_reference_payload(_valid_payload())
    code = "import time; time.sleep(30)"

    with pytest.raises(ReferenceExecutionError, match="exceeded timeout"):
        run_command(fixture, [sys.executable, "-c", code], timeout_s=0.05)


@pytest.mark.skipif(sys.platform != "linux", reason="liveness assertion uses Linux /proc")
def test_json_lines_command_kills_descendants_after_success(tmp_path) -> None:
    fixture = validate_reference_payload(_valid_payload())
    pid_path = tmp_path / "child.pid"
    code = f"""
import json
from pathlib import Path
import subprocess
import sys

child = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(30)"],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
Path(sys.argv[1]).write_text(str(child.pid), encoding="ascii")
for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps({{"format": {RESPONSE_FORMAT!r}, "case": request["case"], "logpdf": -2.0}}))
"""
    child_pid: int | None = None
    try:
        run_command(fixture, [sys.executable, "-c", code, str(pid_path)])
        child_pid = int(pid_path.read_text(encoding="ascii"))

        deadline = time.monotonic() + 1.0
        while _process_is_running(child_pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not _process_is_running(child_pid)
    finally:
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _process_is_running(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text(encoding="ascii").split()[2]
    except (FileNotFoundError, ProcessLookupError):
        return False
    return state != "Z"


def test_json_lines_command_rejects_invalid_utf8() -> None:
    fixture = validate_reference_payload(_valid_payload())
    code = "import sys; sys.stdout.buffer.write(bytes([255])); sys.stdout.buffer.flush()"

    with pytest.raises(ReferenceProtocolError, match="UTF-8"):
        run_command(fixture, [sys.executable, "-c", code])


def test_committed_fixture_covers_boundary_distances_atoms_and_scales() -> None:
    cases = COMMITTED_FIXTURE.density_cases
    powers = {case.p for case in cases}
    expected_powers = {
        1.000001,
        1.0001,
        1.01,
        1.25,
        1.5,
        1.75,
        1.99,
        1.9999,
        1.999999,
    }

    assert expected_powers <= powers
    assert any(case.y == 0.0 for case in cases)
    assert any(case.y > 0.0 for case in cases)
    assert any(case.weight != 1.0 for case in cases)
    assert max(case.mu for case in cases) / min(case.mu for case in cases) >= 1e9
    assert max(case.phi for case in cases) / min(case.phi for case in cases) >= 1e9
    assert COMMITTED_FIXTURE.tolerances.logpdf_abs <= 2.5e-9
    assert COMMITTED_FIXTURE.tolerances.p_abs <= 2e-4
    assert COMMITTED_FIXTURE.tolerances.phi_rel <= 5e-4

    profile = COMMITTED_FIXTURE.profile_cases[0]
    assert profile.recipe == "normal_log_mean.v1"
    assert profile.reference_converged
    assert 1.0 < profile.reference_p < 2.0
    assert profile.reference_phi > 0.0
    assert (
        profile.response_sha256
        == "7d2c5cf30a0d8f3c1a7fb281adb2c864900f1ec16e59fdfff536d197f3186477"
    )


@pytest.mark.parametrize(
    "profile",
    COMMITTED_FIXTURE.profile_cases,
    ids=lambda profile: profile.case,
)
def test_committed_profile_recipe_regenerates_response_digest(profile) -> None:
    _, y = _regenerate_profile_response(profile)

    assert _response_sha256(y) == profile.response_sha256


@pytest.mark.slow
@pytest.mark.parametrize(
    "profile",
    COMMITTED_FIXTURE.profile_cases,
    ids=lambda profile: profile.case,
)
def test_public_profile_matches_joint_neutral_reference(profile) -> None:
    x, y = _regenerate_profile_response(profile)
    assert _response_sha256(y) == profile.response_sha256

    model = SuperGLM(
        family=Tweedie(1.5),
        selection_penalty=0.0,
        features={"x": Numeric()},
    )
    result = estimate_tweedie_p(
        model,
        pd.DataFrame({"x": x}),
        y,
        phi_method="mle",
        method="brent",
    )

    assert result.p_hat == pytest.approx(
        profile.reference_p,
        rel=0.0,
        abs=COMMITTED_FIXTURE.tolerances.p_abs,
    )
    assert result.phi_hat == pytest.approx(
        profile.reference_phi,
        rel=COMMITTED_FIXTURE.tolerances.phi_rel,
        abs=0.0,
    )
    assert result.converged is profile.reference_converged
    assert result.outer_converged
    assert result.outer_boundary is None
    assert result.fit_converged
    assert result.solver_converged
    assert result.objective_finite
    assert result.phi_converged
    assert not result.phi_used_fallback
    assert result.phi_n_fallback_evaluations == 0
    assert result.phi_n_value_only_evaluations == 0
    assert result.density_exact is True
    assert result.density_method == "exact"
    assert result.n_saddlepoint == 0


@pytest.mark.slow
def test_certified_density_work_is_bounded_for_ordinary_positive_cases() -> None:
    n = 200
    mu = np.geomspace(0.05, 20.0, n)
    y = mu * np.exp(np.linspace(-1.5, 1.5, n))
    weights = np.geomspace(0.5, 2.0, n)

    for power in (1.05, 1.5, 1.95):
        evaluation = evaluate_tweedie_density(
            y,
            mu,
            0.8,
            power,
            weights=weights,
        )
        diagnostics = evaluation.diagnostics

        assert diagnostics.exact
        assert diagnostics.certified
        assert diagnostics.n_positive == n
        assert diagnostics.n_exact == n
        assert diagnostics.n_approximate == 0
        assert diagnostics.max_terms < 100_000


@pytest.mark.parametrize(
    "case",
    COMMITTED_FIXTURE.density_cases,
    ids=lambda case: case.case,
)
def test_public_density_matches_neutral_reference_fixture(case) -> None:
    y = np.array([case.y])
    mu = np.array([case.mu])
    weights = np.array([case.weight])
    before = (y.copy(), mu.copy(), weights.copy())

    result = tweedie_logpdf(y, mu, case.phi, case.p, weights=weights)

    assert result[0] == pytest.approx(
        case.logpdf,
        rel=0.0,
        abs=COMMITTED_FIXTURE.tolerances.logpdf_abs,
    )
    np.testing.assert_array_equal(y, before[0])
    np.testing.assert_array_equal(mu, before[1])
    np.testing.assert_array_equal(weights, before[2])


def test_checker_cli_self_check_reports_strict_maximum_error() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(CHECKER_PATH),
            "--fixture",
            str(FIXTURE_PATH),
            "--self-check",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30.0,
    )

    assert completed.returncode == 0, completed.stderr
    assert f"checked={len(COMMITTED_FIXTURE.density_cases)}" in completed.stdout
    assert "max_local_fixture_abs=" in completed.stdout
    assert completed.stderr == ""


def test_checker_cli_reports_uncertifiable_fixture_without_traceback(tmp_path) -> None:
    payload = _valid_payload()
    payload["density_cases"][0].update(  # type: ignore[index]
        y=1.0,
        mu=1.0,
        phi=float(np.nextafter(0.0, 1.0)),
        p=1.6,
        weight=1.0,
        logpdf=0.0,
    )
    fixture_path = tmp_path / "uncertifiable.json"
    fixture_path.write_text(json.dumps(payload), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(CHECKER_PATH), "--fixture", str(fixture_path), "--self-check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30.0,
    )

    assert completed.returncode == ReferenceComparisonError.exit_code
    assert "zero_atom could not certify" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert completed.stdout == ""
