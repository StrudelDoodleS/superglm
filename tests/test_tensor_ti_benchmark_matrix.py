import benchmarks.benchmark_tensor_ti_freq as bench
import numpy as np
import pandas as pd
from benchmarks.benchmark_tensor_ti_freq import (
    ROOT,
    FitControls,
    build_case_deltas,
    build_fairness_cases,
    build_superglm_cases,
    build_superglm_control_profiles,
    select_named_items,
    thread_control_metadata,
)


def test_superglm_tensor_benchmark_case_matrix_includes_scaling_cases():
    cases = build_superglm_cases()
    names = [case.name for case in cases]

    assert names[:2] == ["baseline_discrete", "baseline_plus_ti_discrete"]
    assert names == [
        "baseline_discrete",
        "baseline_plus_ti_discrete",
        "baseline_plus_2_tensors_discrete",
        "baseline_plus_3_tensors_discrete",
        "baseline_plus_spline_cat_discrete",
        "baseline_plus_2_spline_cat_discrete",
        "baseline_plus_mixed_tensor_spline_cat_discrete",
    ]


def test_superglm_tensor_benchmark_case_deltas_keep_legacy_one_tensor_delta():
    baseline = {"model": "baseline_discrete", "fit_s": 1.0, "gini_model": 0.2}
    one_tensor = {
        "model": "baseline_plus_ti_discrete",
        "fit_s": 3.5,
        "gini_model": 0.25,
    }
    two_tensor = {
        "model": "baseline_plus_2_tensors_discrete",
        "fit_s": 7.0,
        "gini_model": 0.27,
    }

    deltas = build_case_deltas([baseline, one_tensor, two_tensor])

    assert deltas["fit_s"] == 2.5
    assert deltas["gini_model"] == 0.04999999999999999
    assert deltas["by_case"]["baseline_plus_ti_discrete"]["fit_s"] == 2.5
    assert deltas["by_case"]["baseline_plus_2_tensors_discrete"]["fit_s"] == 6.0


def test_mgcv_comparator_declares_matching_scaling_cases():
    script = (ROOT / "benchmarks" / "benchmark_tensor_ti_mgcv.R").read_text()
    for case in build_superglm_cases():
        expected_name = case.name.replace("baseline", "mgcv_baseline")
        assert expected_name in script


def test_superglm_fairness_profiles_record_strict_and_candidate_controls():
    profiles = build_superglm_control_profiles()
    by_name = {profile.name: profile for profile in profiles}

    assert list(by_name) == [
        "S0_current_default",
        "S1_strict",
        "S2_mgcv_ish",
        "S3_practical",
        "S4_relaxed_candidate",
        "S5_very_relaxed_candidate",
    ]
    assert by_name["S1_strict"].pirls_tol == 1e-7
    assert by_name["S1_strict"].reml_tol == 1e-7
    assert by_name["S1_strict"].runtime_validation == "full"
    assert by_name["S4_relaxed_candidate"].interaction_mode == "fast_candidate"
    assert by_name["S5_very_relaxed_candidate"].fit_kwargs()["reml_tol"] == 1e-4


def test_superglm_fairness_cases_cover_baseline_and_one_tensor_only():
    assert [case.name for case in build_fairness_cases()] == [
        "baseline_discrete",
        "baseline_plus_ti_discrete",
    ]


def test_select_named_items_filters_profiles_by_comma_separated_names():
    profiles = build_superglm_control_profiles()

    selected = select_named_items(profiles, "S1_strict,S3_practical")

    assert [profile.name for profile in selected] == ["S1_strict", "S3_practical"]


def test_fit_case_return_eta_bypasses_multiprocessing_queue(monkeypatch):
    def fake_fit_case_result(*args, **kwargs):
        return {
            "model": args[0],
            "fit_s": 0.1,
            "timed_out": False,
            "_eta_test": np.array([1.0, 2.0]),
        }

    def fail_get_context(*args, **kwargs):  # pragma: no cover - only used on failure
        raise AssertionError("return_eta=True should not enter multiprocessing path")

    monkeypatch.setattr(bench, "_fit_case_result", fake_fit_case_result)
    monkeypatch.setattr(bench.mp, "get_context", fail_get_context)

    row = bench.fit_case(
        "baseline_discrete",
        pd.DataFrame({"x": [1.0]}),
        pd.DataFrame({"x": [1.0]}),
        np.array([0.0]),
        np.array([0.0]),
        np.array([1.0]),
        np.array([1.0]),
        interactions=(),
        controls=FitControls("test"),
        return_eta=True,
    )

    assert row["_eta_test"].tolist() == [1.0, 2.0]


def test_case_deltas_include_deviance_and_fit_controls():
    baseline = {
        "model": "baseline_discrete",
        "fit_s": 1.0,
        "gini_model": 0.2,
        "deviance": 10.0,
        "control": {"pirls_tol": 1e-6},
    }
    one_tensor = {
        "model": "baseline_plus_ti_discrete",
        "fit_s": 3.5,
        "gini_model": 0.25,
        "deviance": 8.0,
        "control": {"pirls_tol": 1e-6},
    }

    deltas = build_case_deltas([baseline, one_tensor])

    assert deltas["deviance"] == -2.0
    assert deltas["by_case"]["baseline_plus_ti_discrete"]["deviance"] == -2.0


def test_thread_control_metadata_records_expected_environment_keys(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")

    metadata = thread_control_metadata()

    assert metadata["OMP_NUM_THREADS"] == "1"
    assert metadata["OPENBLAS_NUM_THREADS"] == "1"
    assert "MKL_NUM_THREADS" in metadata


def test_mgcv_comparator_declares_fairness_controls_and_threads():
    script = (ROOT / "benchmarks" / "benchmark_tensor_ti_mgcv.R").read_text()

    for label in ("default", "explicit_default_like", "loose_1e_6", "relaxed_1e_5"):
        assert label in script
    assert "gam.control(epsilon = 1e-7, mgcv.tol = 1e-7)" in script
    assert "gam.control(epsilon = 1e-6, mgcv.tol = 1e-6)" in script
    assert "nthreads = 1" in script
