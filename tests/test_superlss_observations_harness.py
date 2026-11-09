"""Runner contracts, not another family/kernel test suite.

These witnesses fail if failures are discarded, identities are unchecked, or a
backend/numerical change is presented as a comparable complete-fit timing.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks/superlss_observations_dirichlet.py"


@pytest.fixture
def harness():
    assert SCRIPT.is_file(), "executable observation harness has not been implemented"
    spec = importlib.util.spec_from_file_location("observation_harness", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def record(harness, tmp_path, name, *, role="candidate"):
    arrays = {"prediction": np.array([1.0, 2.0]), "covariance": np.eye(2)}
    return {
        "session": "same-session",
        "role": role,
        "status": "completed",
        "source": {"sha": harness.BASELINE if role == "baseline" else "candidate-head"},
        "identity": dict.fromkeys(
            ("data", "weights", "spec", "environment", "machine", "threads", "harness"), "same"
        ),
        "observations": [
            {
                "kind": "warm",
                "seconds": 2.0,
                "status": "completed",
                "converged": True,
                "smoothing_converged": None,
                "backend": "distributional-dense-v1",
                "rank": 2,
                "objective": -3.0,
                "edf": 2.0,
                "condition": 1.0,
                "numerics": harness.save_arrays(tmp_path / f"{name}.npz", arrays),
                "public_status": "completed",
            }
        ],
    }


def test_worker_failure_keeps_partial_first_fit_and_log(harness, tmp_path):
    output = tmp_path / "worker.json"
    partial = {"status": "running", "observations": [{"kind": "first", "seconds": 0.4}]}
    code = (
        "import pathlib,sys; "
        f"pathlib.Path({str(output)!r}).write_text({json.dumps(partial)!r}); "
        "print('deliberate worker failure', file=sys.stderr); sys.exit(7)"
    )
    result = harness.run_process(
        [sys.executable, "-c", code],
        output,
        timeout=10,
        env=os.environ.copy(),
        context={"rows": 20000},
    )
    assert result["status"] == "failed"
    assert result["returncode"] == 7
    assert result["rows"] == 20000
    assert result["observations"] == partial["observations"]
    assert "deliberate worker failure" in Path(result["log"]).read_text()


def test_timeout_is_a_retained_stress_result(harness, tmp_path):
    result = harness.run_process(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        tmp_path / "timeout.json",
        timeout=0.1,
        env=os.environ.copy(),
        context={"case": "tweedie-mixed", "rows": 100000},
    )
    assert result["status"] == "timed_out"
    assert result["rows"] == 100000
    assert result["case"] == "tweedie-mixed"


def test_success_exit_with_missing_or_invalid_receipt_is_failure(harness, tmp_path):
    result = harness.run_process(
        [sys.executable, "-c", "pass"],
        tmp_path / "missing.json",
        timeout=10,
        env=os.environ.copy(),
        context={},
    )
    assert result["status"] == "failed"
    assert "receipt" in result["error"]


def test_baseline_must_be_exact_source_at_approved_head(harness):
    with pytest.raises(ValueError, match="baseline|expected|head"):
        harness.source_identity(ROOT / "src", expected_sha=harness.BASELINE, require_clean=False)
    with pytest.raises(ValueError, match="src|source"):
        harness.source_identity(ROOT / "src/superglm")


@pytest.mark.parametrize(
    "key", ["data", "spec", "weights", "environment", "machine", "threads", "harness"]
)
def test_mismatched_identity_never_produces_timing_ratio(harness, tmp_path, key):
    candidate = record(harness, tmp_path, "candidate")
    baseline = record(harness, tmp_path, "baseline", role="baseline")
    candidate["identity"][key] = "different"
    result = harness.compare_records(candidate, baseline)
    assert not result["comparable"]
    assert key in result["identity_mismatches"]
    assert result["time_ratio"] is None


def test_wrong_baseline_head_is_not_a_control(harness, tmp_path):
    candidate = record(harness, tmp_path, "candidate")
    baseline = record(harness, tmp_path, "baseline", role="baseline")
    baseline["source"]["sha"] = "wrong"
    result = harness.compare_records(candidate, baseline)
    assert "baseline_source" in result["identity_mismatches"]
    assert result["time_ratio"] is None


def test_backend_change_is_separate_from_numerical_equality(harness, tmp_path):
    candidate = record(harness, tmp_path, "candidate")
    baseline = record(harness, tmp_path, "baseline", role="baseline")
    candidate["observations"][0]["backend"] = "distributional-chunked-v1"
    result = harness.compare_records(candidate, baseline)
    assert not result["comparable"]
    assert result["numerical_match"]
    assert not result["backend_match"]
    assert result["time_ratio"] is None


def test_numerical_mismatch_has_delta_and_no_accepted_ratio(harness, tmp_path):
    candidate = record(harness, tmp_path, "candidate")
    baseline = record(harness, tmp_path, "baseline", role="baseline")
    candidate["observations"][0]["numerics"] = harness.save_arrays(
        tmp_path / "changed.npz", {"prediction": np.array([1.0, 2.25]), "covariance": np.eye(2)}
    )
    result = harness.compare_records(candidate, baseline)
    assert not result["comparable"]
    assert result["deltas"]["prediction"]["max_absolute"] == 0.25
    assert result["deltas"]["prediction"]["tolerance"] < 0.25
    assert result["time_ratio"] is None


def test_nonconvergence_and_missing_public_lifecycle_remain_visible(harness, tmp_path):
    candidate = record(harness, tmp_path, "candidate")
    baseline = record(harness, tmp_path, "baseline", role="baseline")
    candidate["observations"][0].update(converged=False, public_status="missing_stage")
    result = harness.compare_records(candidate, baseline)
    assert not result["comparable"]
    assert not result["convergence_match"]
    assert result["time_ratio"] is None
    assert candidate["observations"][0]["public_status"] == "missing_stage"


def test_array_artifact_tampering_is_detected(harness, tmp_path):
    candidate = record(harness, tmp_path, "candidate")
    baseline = record(harness, tmp_path, "baseline", role="baseline")
    # Still a valid NPZ with identical arrays: integrity, not a load/numerical
    # failure, must catch this replacement of a recorded artifact.
    np.savez(
        candidate["observations"][0]["numerics"]["path"],
        prediction=np.array([1.0, 2.0]),
        covariance=np.eye(2),
    )
    result = harness.compare_records(candidate, baseline)
    assert not result["comparable"]
    assert "artifact" in result["error"]


def test_equal_sources_and_outputs_allow_control_drift_ratio(harness, tmp_path):
    before = record(harness, tmp_path, "before", role="baseline")
    after = record(harness, tmp_path, "after", role="baseline")
    after["observations"][0]["seconds"] = 2.2
    result = harness.compare_records(after, before)
    assert result["comparable"]
    assert result["time_ratio"] == pytest.approx(1.1)


def test_cross_session_wall_times_are_not_same_session_controls(harness, tmp_path):
    candidate = record(harness, tmp_path, "candidate")
    baseline = record(harness, tmp_path, "baseline", role="baseline")
    candidate["session"] = "another-session"
    result = harness.compare_records(candidate, baseline)
    assert not result["comparable"]
    assert "session" in result["identity_mismatches"]
    assert result["time_ratio"] is None


def test_untimed_dense_discrete_equivalence_does_not_require_equal_backends(harness, tmp_path):
    dense = record(harness, tmp_path, "dense")
    discrete = record(harness, tmp_path, "discrete")
    discrete["observations"][0]["backend"] = "distributional-chunked-v1"
    result = harness.compare_records(discrete, dense, execution_check=False)
    assert result["comparable"]
    assert result["numerical_match"]
    assert not result["backend_match"]
    assert result["time_ratio"] is None


def test_reference_dirichlet_joint_density_and_frequency_are_independent(harness):
    case = harness.Case("dirichlet", components=3, weights="frequency")
    data = {
        "y": np.array([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]]),
        "lower": np.full(2, -np.inf),
        "upper": np.full(2, np.inf),
        "side": np.zeros(2),
        "semantics": "frequency",
        "weights": np.array([2.0, 3.0]),
    }
    # Dirichlet(1,1,1) has constant density Gamma(3)=2 on the simplex.
    np.testing.assert_allclose(
        harness.reference_rows(case, data, np.ones(3)), np.log(2) * np.array([2, 3]), rtol=1e-14
    )


def _stub_runner(harness, monkeypatch, tmp_path, make_result):
    monkeypatch.setattr(harness, "ROOT", tmp_path)
    monkeypatch.setattr(
        harness,
        "source_identity",
        lambda *a, **k: {"path": str(tmp_path / "src"), "sha": "candidate-head"},
    )
    monkeypatch.setattr(
        harness,
        "run_process",
        lambda *a, context, **k: {**context, **make_result(context)},
    )
    return tmp_path / "runner.json"


@pytest.mark.parametrize("public_status", ["missing_stage", "failed", "completed"])
def test_reference_only_runner_checks_nested_public_status(
    harness, monkeypatch, tmp_path, public_status
):
    result = {
        "status": "completed",
        "observations": [],
        "reference": {
            "status": "matched",
            "production": {"public_status": public_status, "converged": True},
        },
    }
    output = _stub_runner(harness, monkeypatch, tmp_path, lambda context: result)
    code = harness.main(
        ["--case", "dirichlet-k3-intercept", "--phases", "reference", "--output", str(output)]
    )
    receipt = json.loads(output.read_text())
    assert (code != 0) == (public_status != "completed")
    assert receipt["runner_status"] == (
        "executed" if public_status == "completed" else "attention_required"
    )
    assert receipt["runs"][0]["reference"]["production"]["public_status"] == public_status


@pytest.mark.parametrize(
    "backend", ["distributional-dense-v1", "distributional-chunked-v1", "correct"]
)
def test_equivalence_runner_requires_both_observed_backends(
    harness, monkeypatch, tmp_path, backend
):
    def make_result(context):
        result = record(harness, tmp_path, context["id"])
        observed = (
            backend
            if backend != "correct"
            else (
                "distributional-chunked-v1"
                if context["config"]["discrete"]
                else "distributional-dense-v1"
            )
        )
        result["observations"][0].update(kind="validation", backend=observed)
        return result

    output = _stub_runner(harness, monkeypatch, tmp_path, make_result)
    code = harness.main(
        [
            "--case",
            "gaussian-offset-arrays",
            "--phases",
            "reference",
            "--equivalence",
            "--output",
            str(output),
        ]
    )
    receipt = json.loads(output.read_text())
    assert (code != 0) == (backend != "correct")
    assert receipt["runner_status"] == (
        "executed" if backend == "correct" else "attention_required"
    )
    assert all(c["numerical_match"] for c in receipt["comparisons"])
    assert all(
        c["backend_coverage"]["match"] == (backend == "correct") for c in receipt["comparisons"]
    )
    assert all(c["time_ratio"] is None for c in receipt["comparisons"])


def _dirichlet_public_stub(prediction):
    from types import SimpleNamespace

    import pandas as pd
    from scipy.stats import dirichlet

    frame = pd.DataFrame({"x": [0.0]}, index=[17])
    alpha = pd.DataFrame(
        [[1.0, 2.0, 3.0]], columns=["alpha_0", "alpha_1", "alpha_2"], index=frame.index
    )
    model = SimpleNamespace(
        family=SimpleNamespace(response_covariance=lambda a: np.zeros((1, 3, 3))),
        predict_parameters=lambda *a, **k: alpha,
        predict_link=lambda *a, **k: (np.log(alpha), alpha * 0),
        predict=lambda *a, **k: pd.DataFrame(
            [prediction], columns=["a", "b", "c"], index=frame.index
        ),
        scores=lambda *a, **k: pd.DataFrame(
            {
                "log": -dirichlet.logpdf([0.2, 0.3, 0.5], [1.0, 2.0, 3.0])
                * np.asarray(k["sample_weight"])
            },
            index=frame.index,
        ),
    )
    data = {"columns": ["a", "b", "c"], "y": np.array([[0.2, 0.3, 0.5]]), "weights": np.ones(1)}
    return model, frame, data


def test_dirichlet_public_prediction_must_equal_normalized_alpha(harness):
    model, frame, data = _dirichlet_public_stub([1 / 3, 1 / 3, 1 / 3])
    with pytest.raises(ValueError, match="Dirichlet.*mean") as caught:
        harness.public_outputs(
            model, frame, harness.Case("dirichlet", components=3), data, "candidate"
        )
    assert (
        caught.value.checks["dirichlet_mean_delta"]
        > caught.value.checks["dirichlet_mean_tolerance"]
    )


def test_dirichlet_public_normalized_alpha_mean_passes(harness):
    model, frame, data = _dirichlet_public_stub([1 / 6, 1 / 3, 1 / 2])
    _, checks = harness.public_outputs(
        model, frame, harness.Case("dirichlet", components=3), data, "candidate"
    )
    assert checks["simplex_error"] == 0.0
    assert checks["dirichlet_mean_delta"] == 0.0


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("corruption", ["sign", "shift"])
def test_dirichlet_public_scores_have_an_independent_density_guard(harness, discrete, corruption):
    model, frame, data = _dirichlet_public_stub([1 / 6, 1 / 3, 1 / 2])
    model.discrete = discrete
    original = model.scores

    def wrong(*args, **kwargs):
        scores = original(*args, **kwargs)
        return -scores if corruption == "sign" else scores + 0.5

    model.scores = wrong
    with pytest.raises(harness.PublicOutputError, match="Dirichlet log scores"):
        harness.public_outputs(
            model, frame, harness.Case("dirichlet", components=3), data, "candidate"
        )


def test_frequency_weighted_public_scores_pass_the_independent_guard(harness):
    model, frame, data = _dirichlet_public_stub([1 / 6, 1 / 3, 1 / 2])
    data["weights"] *= 3
    _, checks = harness.public_outputs(
        model,
        frame,
        harness.Case("dirichlet", weights="frequency", components=3),
        data,
        "candidate",
    )
    assert (
        checks["dirichlet_log_score_reference_max_delta"]
        <= checks["dirichlet_log_score_reference_tolerance"]
    )


def test_binned_training_and_continuous_score_identity_is_not_asserted(harness):
    checks = {"log_score_sum": -2.0}
    harness.check_score_identity(checks, 3.0, discrete=True)
    assert checks["log_score_likelihood_delta"] == 1.0
    assert checks["log_score_likelihood_identity_applicable"] is False


def test_exact_training_score_identity_still_rejects_wrong_values(harness):
    checks = {"log_score_sum": -2.0}
    with pytest.raises(harness.PublicOutputError, match="fitted normalized likelihood"):
        harness.check_score_identity(checks, 3.0, discrete=False)
    assert checks["log_score_likelihood_identity_applicable"] is True


def test_candidate_offset_missing_stage_never_falls_back_to_fit_mapping(harness, monkeypatch):
    import superglm

    monkeypatch.setattr(superglm, "bind_predictor", lambda family, name: None)
    case = harness.CASES["gaussian-offset-arrays"]
    with pytest.raises(RuntimeError, match="Task9"):
        harness.offset_adapter(case, "candidate")
    assert harness.offset_adapter(case, "baseline").startswith("baseline-1b63-fit-mapping")


def test_retained_views_count_buffer_once_even_if_cache_path_is_later(harness):
    owner = np.arange(100.0)
    result = harness.retained_memory({"data": owner[:30], "prepared": {"log_shares": owner[30:]}})
    assert result["unique_buffer_bytes"] == owner.nbytes
    assert result["cache_buffer_bytes"] == owner.nbytes


def test_cli_lists_required_cells_without_fitting(harness):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--list-cases"], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    for name in (
        "tweedie-mixed",
        "negative-binomial-combined-prior",
        "two-piece-log-normal-mean-complete",
        "dirichlet-k5-smooth",
        "gaussian-offset-arrays",
    ):
        assert name in result.stdout


def test_data_identity_is_reproducible_and_weights_are_distinct(harness):
    case = harness.CASES["negative-binomial-combined-prior"]
    a = harness.make_data(case, 48)
    b = harness.make_data(case, 48)
    assert a["identity"] == b["identity"]
    other = copy.deepcopy(a)
    other["weights"][0] += 1
    assert harness.data_identity(other)["weights"] != a["identity"]["weights"]


def test_bounded_runner_uses_public_censor_labels(harness):
    case = harness.CASES["gaussian-bounded"]
    data = {"side": np.array([-1, 0, 1]), "lower": np.full(3, -2.0), "upper": np.full(3, 2.0)}
    options = harness.bound_options(case, data)
    # Public fit/scoring deliberately refuse internal integer side codes.
    assert list(options["censored"]) == ["left", "none", "right"]


def test_memory_includes_retained_pandas_numpy_blocks(harness):
    import pandas as pd

    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "z": [4.0, 5.0, 6.0]})
    result = harness.retained_memory({"training_frame": frame})
    assert result["unique_buffer_bytes"] == 6 * 8


def test_dirichlet_array_prediction_is_staged_not_replaced_by_internal_means(harness):
    from types import SimpleNamespace

    import pandas as pd

    frame = pd.DataFrame({"x": [0.0, 1.0]})
    alpha = pd.DataFrame(np.ones((2, 3)), columns=["alpha_0", "alpha_1", "alpha_2"])
    model = SimpleNamespace(
        family=object(),
        predict_parameters=lambda frame: alpha,
        predict_link=lambda frame, se_fit: (alpha, alpha),
        predict=lambda frame: np.full((2, 3), 1 / 3),
    )
    with pytest.raises(RuntimeError, match="Task6.*DataFrame"):
        harness.public_outputs(model, frame, harness.CASES["dirichlet-k3-fixed"], {}, "candidate")


@pytest.mark.parametrize(
    ("law", "theta", "response", "lower", "upper", "expected"),
    [
        ("negative-binomial", [2.0, 3.0], 1.0, 0.0, 1.0, 0.0),
        ("tweedie", [1.0, 1.0, 1.5], 0.0, -np.inf, np.inf, -4.0),
        ("two-piece-normal", [0.0, 1.0, 0.0], 0.0, -np.inf, np.inf, -np.log(2 * np.pi)),
    ],
)
def test_small_reference_singleton_atom_and_normal_limits(
    harness, law, theta, response, lower, upper, expected
):
    data = dict(
        y=np.array([response]),
        lower=np.array([lower]),
        upper=np.array([upper]),
        side=np.array([0]),
        weights=np.array([2.0]),
        semantics="frequency",
    )
    actual = harness.reference_rows(harness.Case(law), data, np.array(theta))
    np.testing.assert_allclose(actual, [expected], atol=2e-14, rtol=2e-14)


def test_profile_filters_package_source_not_checkout_directory_name(harness):
    import cProfile

    namespace = {}
    for name, filename in (
        ("kernel_call", "/tmp/superglm/tree/src/superglm/distributional/kernels/law.py"),
        ("dependency_call", "/tmp/superglm/tree/.venv/site-packages/dependency.py"),
    ):
        exec(compile(f"def {name}(): return 1", filename, "exec"), namespace)
    with cProfile.Profile() as profiler:
        namespace["kernel_call"]()
        namespace["dependency_call"]()
    summary = harness.profile_summary(profiler)
    assert len(summary["top"]) == 1
    assert summary["dispatch"][0]["function"].startswith("superglm/distributional/kernels/")


def test_nb_reference_censor_numerator_does_not_change_truncation_law(harness):
    # Prior w=2, mean=2, theta=1 => NB(size=2,p=1/3), observed as count/2.
    # On (0,1.5], P(count=1,2,3)=(36,36,32)/243. Right at .5 keeps
    # counts 2,3; left at 1 keeps counts 1,2. No production oracle is involved.
    data = dict(
        y=np.array([0.5, 1.0]),
        lower=np.zeros(2),
        upper=np.full(2, 1.5),
        side=np.array([1, -1]),
        weights=np.full(2, 2.0),
        semantics="prior",
    )
    actual = harness.reference_rows(harness.Case("negative-binomial"), data, np.array([2.0, 1.0]))
    np.testing.assert_allclose(actual, np.log([17 / 26, 9 / 13]), atol=2e-14, rtol=2e-14)
