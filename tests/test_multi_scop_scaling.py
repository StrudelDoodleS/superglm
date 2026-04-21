def test_build_scenarios_covers_discrete_matrix_and_exact_spot_checks():
    from benchmarks.multi_scop_scaling import build_scenarios

    scenarios = build_scenarios()
    names = {(s.mode, s.n_constrained) for s in scenarios}
    assert ("discrete", 1) in names
    assert ("discrete", 16) in names
    assert ("exact", 1) in names
    assert ("exact", 2) in names
    assert ("exact", 4) in names


def test_summarize_lambda_activity_counts_floor_pinned_and_active_terms():
    from benchmarks.multi_scop_scaling import summarize_lambda_activity

    lambdas = {"x1": 1e-4, "x2": 0.05, "x3": 0.2}
    summary = summarize_lambda_activity(lambdas, floor=1e-4, active_threshold=1e-3)
    assert summary["n_floor"] == 1
    assert summary["n_active"] == 2
