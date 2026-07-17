from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import benchmarks.benchmark_fit_state_trace as benchmark_module
import numpy as np
import pytest
import tabmat
from benchmarks.benchmark_fit_state_trace import (
    CASES,
    _build_parser,
    _comparison_report,
    _count_tabmat_kernel_calls,
    _execution_order,
    _matrix_backend_metadata,
    _run_worker_case,
    _strip_fidelity_vectors,
    _summarize_samples,
    _validate_comparison_context,
    _validate_suite_quality,
    _validate_worker_record,
    _version_metadata,
    _worker_environment,
    compare_runs,
)

REQUIRED_CASES = {
    "dense_fit",
    "categorical_fit",
    "spline_fit",
    "exact_reml",
    "discrete_reml",
    "compact_reml",
}


def _record(case: str, repeat: int, wall_time_s: float) -> dict[str, object]:
    return {
        "case": case,
        "repeat": repeat,
        "order": repeat,
        "wall_time_s": wall_time_s,
        "python_peak_bytes": 1_000 + repeat,
        "rss_before_fit_bytes": 1_500 + repeat,
        "rss_peak_bytes": 2_000 + repeat,
        "rss_peak_delta_bytes": 500,
        "deviance": 10.0,
        "effective_df": 2.0,
        "prediction_checksum": 3.0,
        "prediction_projection": 4.0,
        "prediction_l2": 5.0,
        "prediction_values": [1.0, 2.0],
        "beta_values": [0.25, -0.5],
        "intercept": 0.75,
        "phi": 1.0,
        "log_likelihood": -10.0,
        "null_log_likelihood": -20.0,
        "null_deviance": 20.0,
        "explained_deviance": 0.5,
        "pearson_chi2": 12.0,
        "n_obs": 2,
        "reml_objective": None,
        "lambda_values": {},
        "n_iter": 4,
        "converged": True,
        "reml_converged": None,
        "overall_converged": True,
        "reml_diagnostics": None,
        "profile": {},
        "matrix_backend": {
            "tabmat_built": True,
            "tabmat_prepared": True,
            "tabmat_split_type": "SplitMatrix",
            "tabmat_component_types": ["DenseMatrix"],
        },
        "tabmat_kernel_calls": {"sandwich": 1, "transpose_matvec": 2},
    }


def test_transaction_benchmark_covers_required_paths() -> None:
    assert REQUIRED_CASES <= set(CASES)


def test_categorical_fixture_can_build_a_tabmat_categorical_component() -> None:
    prepared = CASES["categorical_fit"](0.02)
    prepared.model._build_design_matrix(
        prepared.X,
        prepared.y,
        prepared.sample_weight,
        prepared.offset,
    )
    prepared.model._dm.tabmat_split

    backend = _matrix_backend_metadata(prepared.model)

    assert backend["tabmat_built"] is True
    assert backend["tabmat_prepared"] is True
    assert backend["tabmat_split_type"] == "SplitMatrix"
    assert "CategoricalMatrix" in backend["tabmat_component_types"]
    assert prepared.model._direct_solve == "gram"


def test_tabmat_kernel_counter_records_actual_split_calls() -> None:
    split = tabmat.SplitMatrix([tabmat.DenseMatrix(np.ones((3, 1)))])

    with _count_tabmat_kernel_calls() as calls:
        split.sandwich(np.ones(3))
        split.transpose_matvec(np.ones(3))

    assert calls == {"sandwich": 1, "transpose_matvec": 1}


def test_categorical_fit_dispatches_prepared_tabmat_kernels() -> None:
    prepared = CASES["categorical_fit"](0.02)

    with _count_tabmat_kernel_calls() as calls:
        prepared.model.fit(prepared.X, prepared.y)

    assert calls["sandwich"] > 0
    assert calls["sandwich"] <= calls["transpose_matvec"] <= 2 * calls["sandwich"]
    assert prepared.model._dm._tabmat_centering_candidate is True


def test_dense_fit_does_not_build_an_unused_tabmat_split() -> None:
    prepared = CASES["dense_fit"](0.02)

    prepared.model.fit(prepared.X, prepared.y)

    assert prepared.model._dm._tabmat_built is False
    assert prepared.model._dm._tabmat_centering_candidate is False


def test_low_cardinality_categorical_fit_does_not_build_dense_tabmat_duplicate() -> None:
    rng = np.random.default_rng(160)
    n = 400
    codes = np.resize(np.arange(20), n)
    rng.shuffle(codes)
    X = benchmark_module.pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "category": np.asarray([f"level_{code:02d}" for code in codes], dtype=object),
        }
    )
    y = rng.poisson(1.0, size=n).astype(np.float64)
    model = benchmark_module.SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={
            "x": benchmark_module.Numeric(),
            "category": benchmark_module.Categorical(base="first"),
        },
        direct_solve="gram",
    )

    model.fit(X, y)

    assert model._dm._tabmat_built is False
    assert model._dm._tabmat_centering_candidate is False


def test_categorical_only_fit_keeps_compact_path_without_building_tabmat() -> None:
    rng = np.random.default_rng(161)
    n = 640
    codes = np.resize(np.arange(120), n)
    rng.shuffle(codes)
    X = benchmark_module.pd.DataFrame(
        {"category": np.asarray([f"level_{code:03d}" for code in codes], dtype=object)}
    )
    y = rng.poisson(1.0, size=n).astype(np.float64)
    model = benchmark_module.SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={"category": benchmark_module.Categorical(base="first")},
        direct_solve="gram",
    )

    model.fit(X, y)

    assert model._dm._tabmat_built is False
    assert model._dm._tabmat_centering_candidate is False


def test_categorical_tabmat_fit_is_numerically_deterministic() -> None:
    first = CASES["categorical_fit"](0.02)
    second = CASES["categorical_fit"](0.02)

    first.model.fit(first.X, first.y)
    second.model.fit(second.X, second.y)

    np.testing.assert_allclose(first.model.result.beta, second.model.result.beta, atol=1e-12)
    assert first.model.result.intercept == pytest.approx(second.model.result.intercept, abs=1e-12)
    assert first.model.result.deviance == pytest.approx(second.model.result.deviance, rel=1e-12)


def test_execution_order_alternates_deterministically() -> None:
    names = ("dense_fit", "categorical_fit", "spline_fit")

    assert _execution_order(names, 0) == names
    assert _execution_order(names, 1) == tuple(reversed(names))
    assert _execution_order(names, 2) == names


def test_compare_runs_accepts_numerical_parity() -> None:
    before = _record("dense_fit", 0, 1.0)
    after = dict(before, deviance=float(before["deviance"]) + 1e-12)

    assert compare_runs(before, after, numerical_rtol=1e-10) == []


@pytest.mark.parametrize(
    "key",
    [
        "deviance",
        "effective_df",
        "prediction_checksum",
        "prediction_projection",
        "prediction_l2",
        "phi",
        "intercept",
        "log_likelihood",
        "null_log_likelihood",
        "null_deviance",
        "explained_deviance",
        "pearson_chi2",
    ],
)
def test_compare_runs_rejects_quality_drift(key: str) -> None:
    before = _record("dense_fit", 0, 1.0)
    after = dict(before)
    after[key] = float(after[key]) + 0.01

    failures = compare_runs(before, after, numerical_rtol=1e-8)

    assert any(key in failure for failure in failures)


@pytest.mark.parametrize("key", ["prediction_values", "beta_values"])
def test_compare_runs_rejects_pointwise_or_coefficient_drift(key: str) -> None:
    before = _record("dense_fit", 0, 1.0)
    after = dict(before)
    after[key] = list(before[key])
    after[key][0] += 0.1

    failures = compare_runs(before, after)

    assert any(key in failure for failure in failures)


def test_compare_runs_rejects_observation_count_drift() -> None:
    before = _record("dense_fit", 0, 1.0)
    after = dict(before, n_obs=3)

    failures = compare_runs(before, after)

    assert any("n_obs" in failure for failure in failures)


def test_compare_runs_rejects_reml_and_convergence_drift() -> None:
    before = _record("exact_reml", 0, 1.0)
    before.update(
        reml_objective=12.0,
        lambda_values={"x": 0.5},
        reml_converged=True,
        overall_converged=True,
    )
    after = dict(before, lambda_values={"x": 0.7}, overall_converged=False)

    failures = compare_runs(before, after)

    assert any("lambda_values.x" in failure for failure in failures)
    assert any("overall_converged" in failure for failure in failures)


def test_comparison_rejects_missing_cases() -> None:
    baseline = {"cases": {"dense_fit": _record("dense_fit", 0, 1.0)}}

    with pytest.raises(ValueError, match="case set"):
        _comparison_report(baseline, {})


def test_summarize_samples_reports_medians_and_raw_count() -> None:
    samples = [
        _record("dense_fit", 0, 3.0),
        _record("dense_fit", 1, 1.0),
        _record("dense_fit", 2, 2.0),
        _record("spline_fit", 0, 7.0),
        _record("spline_fit", 1, 5.0),
    ]

    summary = _summarize_samples(samples)

    assert summary["dense_fit"]["sample_count"] == 3
    assert summary["dense_fit"]["median_wall_time_s"] == 2.0
    assert summary["dense_fit"]["median_python_peak_bytes"] == 1_001
    assert summary["dense_fit"]["median_rss_before_fit_bytes"] == 1_501
    assert summary["dense_fit"]["median_rss_peak_bytes"] == 2_001
    assert summary["dense_fit"]["median_rss_peak_delta_bytes"] == 500
    assert summary["dense_fit"]["prediction_values"] == [1.0, 2.0]
    assert summary["dense_fit"]["beta_values"] == [0.25, -0.5]
    assert summary["spline_fit"]["sample_count"] == 2
    assert summary["spline_fit"]["median_wall_time_s"] == 6.0


def test_summarize_samples_requires_reml_convergence() -> None:
    record = _record("exact_reml", 0, 1.0)
    record["reml_converged"] = False
    record["overall_converged"] = False

    summary = _summarize_samples([record])["exact_reml"]

    assert summary["all_converged"] is False
    assert summary["all_reml_converged"] is False


def test_suite_quality_rejects_unconverged_reml_case() -> None:
    summaries = {"exact_reml": _summarize_samples([_record("exact_reml", 0, 1.0)])["exact_reml"]}
    summaries["exact_reml"]["reml_converged"] = False
    summaries["exact_reml"]["overall_converged"] = False

    with pytest.raises(ValueError, match="exact_reml.*converge"):
        _validate_suite_quality(summaries)


def test_raw_timing_samples_strip_duplicate_fidelity_vectors() -> None:
    record = _record("dense_fit", 0, 1.0)

    compact = _strip_fidelity_vectors([record])

    assert "prediction_values" not in compact[0]
    assert "beta_values" not in compact[0]
    assert record["prediction_values"] == [1.0, 2.0]


def test_worker_times_untraced_fit_and_profiles_allocations_separately(monkeypatch) -> None:
    tracing_states: list[bool] = []

    class FakeModel:
        def __init__(self) -> None:
            self.result = SimpleNamespace(
                deviance=1.0,
                effective_df=1.0,
                n_iter=1,
                converged=True,
                phi=1.0,
                beta=np.array([0.25]),
                intercept=0.5,
            )
            self._fit_stats = SimpleNamespace(
                log_likelihood=-1.0,
                null_log_likelihood=-2.0,
                null_deviance=2.0,
                explained_deviance=0.5,
                pearson_chi2=1.0,
                n_obs=80,
            )
            self._reml_result = None
            self._reml_profile = None
            self._dm = None

        def fit(self, X, y, **kwargs) -> None:
            tracing_states.append(__import__("tracemalloc").is_tracing())

        def predict(self, X, offset=None):
            return np.ones(len(X))

    def factory(scale: float):
        n = 80
        return SimpleNamespace(
            model=FakeModel(),
            X=SimpleNamespace(__len__=lambda self: n),
            y=np.ones(n),
            sample_weight=None,
            offset=None,
            fit_method="fit",
            fit_kwargs={},
        )

    # A real sequence supplies a reliable len() while keeping the fake fit inert.
    def sequence_factory(scale: float):
        prepared = factory(scale)
        prepared.X = list(range(80))
        return prepared

    monkeypatch.setitem(CASES, "fake", sequence_factory)

    record = _run_worker_case("fake", repeat=0, order=0, case_scale=1.0)

    assert tracing_states == [False, True]
    assert record["python_peak_bytes"] > 0


def test_scaled_dense_worker_emits_complete_fidelity_record() -> None:
    record = _run_worker_case("dense_fit", repeat=0, order=0, case_scale=0.02)

    assert record["overall_converged"] is True
    assert len(record["prediction_values"]) == 80
    assert len(record["beta_values"]) == 2
    assert record["python_peak_bytes"] > 0


@pytest.mark.parametrize("case", ["exact_reml", "discrete_reml", "compact_reml"])
def test_scaled_reml_workers_converge_with_finite_objective(case: str) -> None:
    record = _run_worker_case(case, repeat=0, order=0, case_scale=0.02)

    assert record["reml_converged"] is True
    assert record["overall_converged"] is True
    assert np.isfinite(record["reml_objective"])
    assert record["reml_diagnostics"]["n_reml_iter"] <= 8


def test_main_returns_nonzero_for_quality_comparison_failure(tmp_path, monkeypatch) -> None:
    output = tmp_path / "candidate.json"
    monkeypatch.setattr(
        benchmark_module,
        "_run_wall_time_suite",
        lambda args: {"comparison": {"dense_fit": {"numerical_failures": ["prediction drift"]}}},
    )

    status = benchmark_module.main(["--output", str(output), "--compare", str(output)])

    assert status == 2


def test_worker_record_validator_accepts_finite_complete_record() -> None:
    _validate_worker_record(_record("dense_fit", 0, 0.1))


@pytest.mark.parametrize("missing", ["deviance", "effective_df", "prediction_checksum"])
def test_worker_record_validator_rejects_missing_numerical_field(missing: str) -> None:
    record = _record("dense_fit", 0, 0.1)
    del record[missing]

    with pytest.raises(ValueError, match=missing):
        _validate_worker_record(record)


def test_worker_record_validator_rejects_nonfinite_numerical_field() -> None:
    record = _record("dense_fit", 0, 0.1)
    record["deviance"] = float("nan")

    with pytest.raises(ValueError, match="deviance"):
        _validate_worker_record(record)


def test_parser_accepts_wall_time_parent_controls(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    compare = tmp_path / "before.json"
    compare.write_text(json.dumps({"cases": {}}), encoding="utf-8")

    args = _build_parser().parse_args(
        [
            "--suite",
            "wall-time",
            "--warmups",
            "2",
            "--repeats",
            "10",
            "--output",
            str(output),
            "--compare",
            str(compare),
        ]
    )

    assert args.suite == "wall-time"
    assert args.warmups == 2
    assert args.repeats == 10
    assert args.output == output
    assert args.compare == compare


def test_parser_accepts_worker_fixture(tmp_path: Path) -> None:
    output = tmp_path / "worker.json"

    args = _build_parser().parse_args(
        ["--worker", "--fixture", "dense_fit", "--output", str(output)]
    )

    assert args.worker is True
    assert args.fixture == "dense_fit"
    assert args.output == output


def test_worker_environment_forces_reproducible_single_thread_limits() -> None:
    environment = _worker_environment()

    assert environment["PYTHONHASHSEED"] == "0"
    assert environment["OMP_NUM_THREADS"] == "1"
    assert environment["OPENBLAS_NUM_THREADS"] == "1"
    assert environment["MKL_NUM_THREADS"] == "1"
    assert environment["NUMBA_NUM_THREADS"] == "1"


def test_version_metadata_identifies_machine_and_git_state() -> None:
    metadata = _version_metadata(environment=_worker_environment())

    assert metadata["machine"]
    assert metadata["os_release"]
    assert "cpu_model" in metadata
    assert isinstance(metadata["git_dirty"], bool)
    assert metadata["pythonhashseed"] == "0"
    assert set(metadata["thread_environment"].values()) == {"1"}


def test_comparison_context_rejects_different_case_scale() -> None:
    metadata = _version_metadata()
    baseline = {
        "schema_version": 1,
        "suite": "wall-time",
        "metadata": metadata,
        "config": {
            "warmups": 2,
            "repeats": 10,
            "case_scale": 1.0,
            "case_names": ["dense_fit"],
            "measurement_contract": {"authoritative": ["wall_time_s"]},
        },
    }
    candidate_config = {
        "warmups": 2,
        "repeats": 10,
        "case_scale": 0.5,
        "case_names": ["dense_fit"],
        "measurement_contract": {"authoritative": ["wall_time_s"]},
    }

    with pytest.raises(ValueError, match="case_scale"):
        _validate_comparison_context(baseline, metadata, candidate_config)


def test_comparison_context_rejects_different_repeat_count() -> None:
    metadata = _version_metadata(environment=_worker_environment())
    baseline = {
        "schema_version": 1,
        "suite": "wall-time",
        "metadata": metadata,
        "config": {
            "warmups": 2,
            "repeats": 10,
            "case_scale": 1.0,
            "case_names": ["dense_fit"],
            "measurement_contract": {"authoritative": ["wall_time_s"]},
        },
    }
    candidate_config = dict(baseline["config"], repeats=5)

    with pytest.raises(ValueError, match="repeats"):
        _validate_comparison_context(baseline, metadata, candidate_config)
