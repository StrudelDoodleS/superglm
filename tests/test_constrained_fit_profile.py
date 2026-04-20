from pathlib import Path

from benchmarks._constrained_fit_profile import (
    ProfileScenario,
    make_synthetic_dataset,
    profile_callstack_and_memory,
    summarize_rows,
    write_profile_artifacts,
)


def test_make_synthetic_dataset_repeated_support_has_fewer_unique_values():
    scenario = ProfileScenario(
        name="single_scop_repeated",
        engine="scop",
        n=1000,
        k=10,
        n_constrained=1,
        repeated_support=True,
        discrete=False,
        use_fremtpl=False,
    )
    X, y, w = make_synthetic_dataset(scenario, seed=42)

    assert len(X) == len(y) == len(w) == 1000
    assert X["x1"].nunique() < len(X)


def test_summarize_rows_emits_expected_columns():
    frame = summarize_rows(
        [
            {
                "scenario": "demo",
                "engine": "scop",
                "mode": "exact",
                "n": 1000,
                "k": 10,
                "n_constrained": 1,
                "runtime_s": 0.5,
                "n_reml_iter": 4,
                "n_pirls_iter": 2,
                "peak_mem_mb": 12.0,
            }
        ]
    )
    assert list(frame.columns) == [
        "scenario",
        "engine",
        "mode",
        "n",
        "k",
        "n_constrained",
        "runtime_s",
        "n_reml_iter",
        "n_pirls_iter",
        "peak_mem_mb",
    ]


def test_write_profile_artifacts_creates_expected_files(tmp_path: Path):
    result, stats_text, memory_stats = profile_callstack_and_memory(lambda: sum(range(10)))

    assert result == 45
    assert "sum" in stats_text
    assert memory_stats["peak_mb"] >= 0.0

    paths = write_profile_artifacts(
        base_dir=tmp_path,
        stem="demo",
        profile_stats={"stats": stats_text},
        memory_stats=memory_stats,
    )
    assert paths["cpu_txt"].exists()
    assert paths["memory_json"].exists()
