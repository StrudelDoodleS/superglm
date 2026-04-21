import numpy as np


def test_build_scenarios_covers_discrete_matrix_and_exact_spot_checks():
    from benchmarks.multi_scop_scaling import build_scenarios

    scenarios = build_scenarios()
    names = {(s.mode, s.n_constrained) for s in scenarios}
    assert ("discrete", 1) in names
    assert ("discrete", 2) in names
    assert ("discrete", 4) in names
    assert ("discrete", 8) in names
    assert ("discrete", 16) in names
    assert ("exact", 1) in names
    assert ("exact", 2) in names
    assert ("exact", 4) in names
    assert ("exact", 8) not in names


def test_summarize_lambda_activity_counts_floor_pinned_and_active_terms():
    from benchmarks.multi_scop_scaling import summarize_lambda_activity

    lambdas = {"x1": 1e-4, "x2": 0.05, "x3": 0.2}
    summary = summarize_lambda_activity(lambdas, floor=1e-4, active_threshold=1e-3)
    assert summary["n_floor"] == 1
    assert summary["n_active"] == 2


def test_make_dataset_builds_repeated_support_constrained_features():
    from benchmarks.multi_scop_scaling import make_dataset

    X, y = make_dataset(n=200, n_constrained=4, seed=7)

    assert list(X.columns) == ["x1", "x2", "x3", "x4"]
    assert X.shape == (200, 4)
    assert y.shape == (200,)
    assert y.dtype == np.float64
    assert all(X[column].nunique() < len(X) for column in X.columns)
    assert np.all((X.to_numpy(dtype=np.float64) >= 0.0) & (X.to_numpy(dtype=np.float64) <= 1.0))


def test_make_dataset_repeats_support_values_for_small_n():
    from benchmarks.multi_scop_scaling import make_dataset

    X, _ = make_dataset(n=10, n_constrained=1, seed=7)

    assert X["x1"].nunique() < len(X)


def test_summarize_rows_uses_run_row_schema_order():
    from benchmarks.multi_scop_scaling import RunRow, summarize_rows

    frame = summarize_rows(
        [
            RunRow(
                mode="discrete",
                n_constrained=4,
                n=100_000,
                k=12,
                runtime_s=1.25,
                peak_mem_mb=128.0,
                converged=True,
                n_reml_iter=5,
                n_pirls_iter=3,
                n_floor=1,
                n_active=3,
            )
        ]
    )

    assert list(frame.columns) == [
        "mode",
        "n_constrained",
        "n",
        "k",
        "runtime_s",
        "peak_mem_mb",
        "converged",
        "n_reml_iter",
        "n_pirls_iter",
        "n_floor",
        "n_active",
    ]
