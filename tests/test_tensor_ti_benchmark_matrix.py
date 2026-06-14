from benchmarks.benchmark_tensor_ti_freq import (
    ROOT,
    build_case_deltas,
    build_superglm_cases,
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
