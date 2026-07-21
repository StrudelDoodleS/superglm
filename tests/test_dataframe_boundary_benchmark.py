from __future__ import annotations

import json
from pathlib import Path

from benchmarks import benchmark_dataframe_boundary as benchmark_module
from benchmarks.benchmark_dataframe_boundary import _compare


def _benchmark_payload(*, rss_bytes: int) -> dict[str, object]:
    raw = [0.010, 0.011, 0.012]
    return {
        "scenarios": {
            "ordinary_scalar_fit": {
                "sample_count": len(raw),
                "raw_wall_time_s": raw,
                "median_wall_time_s": 0.011,
                "mad_wall_time_s": 0.001,
                "median_python_peak_bytes": 2_000_000,
                "median_rss_delta_bytes": rss_bytes,
                "kernel_calls": {"sandwich": 2},
                "matrix": {"group_types": ["DenseGroupMatrix"]},
                "numerical": {"deviance": 1.0},
                "scenario_seed": 2102,
                "dimensions": {"input_rows": 60_000, "input_columns": 16},
            }
        }
    }


def test_comparator_reports_raw_spread_dispatch_and_enforces_rss(
    tmp_path: Path,
    capsys,
) -> None:
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    before_path.write_text(json.dumps(_benchmark_payload(rss_bytes=10_000_000)))
    after_path.write_text(json.dumps(_benchmark_payload(rss_bytes=12_000_000)))

    status = _compare(before_path, after_path, backends=False)
    report = json.loads(capsys.readouterr().out)

    assert status == 1
    assert any("RSS" in failure for failure in report["failures"])
    row = report["comparisons"][0]
    assert row["before_raw_wall_time_s"] == [0.010, 0.011, 0.012]
    assert row["after_raw_wall_time_s"] == [0.010, 0.011, 0.012]
    assert row["before_mad_wall_time_s"] == 0.001
    assert row["after_mad_wall_time_s"] == 0.001
    assert row["before_kernel_calls"] == {"sandwich": 2}
    assert row["after_kernel_calls"] == {"sandwich": 2}
    assert row["before_matrix"] == {"group_types": ["DenseGroupMatrix"]}
    assert row["after_matrix"] == {"group_types": ["DenseGroupMatrix"]}
    assert row["before_numerical"]["deviance"] == 1.0
    assert row["after_numerical"]["deviance"] == 1.0
    assert row["before_scenario_seed"] == 2102
    assert row["after_scenario_seed"] == 2102
    assert row["before_dimensions"] == {"input_rows": 60_000, "input_columns": 16}
    assert row["after_dimensions"] == {"input_rows": 60_000, "input_columns": 16}
    assert row["before_rss_delta_bytes"] == 10_000_000
    assert row["after_rss_delta_bytes"] == 12_000_000


def test_smoke_suite_keeps_one_warmup_and_one_measured_sample(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repeats: list[int] = []

    def fake_worker(name: str, backend: str, scale: float, repeat: int) -> dict[str, object]:
        repeats.append(repeat)
        return {
            "scenario": name,
            "repeat": repeat,
            "wall_time_s": 0.01,
            "python_peak_bytes": 100,
            "rss_delta_bytes": 100,
            "kernel_calls": {},
            "matrix": {},
            "numerical": {},
            "scenario_seed": 2102,
            "dimensions": {"input_rows": 1, "input_columns": 1},
        }

    monkeypatch.setattr(benchmark_module, "_run_worker", fake_worker)
    args = benchmark_module._parser().parse_args(
        [
            "--smoke",
            "--scenario",
            "ordinary_scalar_fit",
            "--output",
            str(tmp_path / "smoke.json"),
        ]
    )

    payload = benchmark_module._run_suite(args)

    assert repeats == [-1, 0]
    assert payload["config"]["warmups"] == 1
    assert payload["config"]["repeats"] == 1
