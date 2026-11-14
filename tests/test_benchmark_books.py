"""benchmarks/books.py: a counts-only record carries every field, and compare refuses a mismatched key.

The harness's numbers stand in for re-timing master in pull request bodies, so
its records must say what was measured and it must refuse a baseline whose key
(dataset, spec, environment, machine, thread setting) differs from the branch's.
"""

from __future__ import annotations

import copy
import json
import os
from types import SimpleNamespace

import pytest
from benchmarks import books


@pytest.fixture(scope="module")
def run():
    return books.run_case("smoke_poisson", "pinned", counts_only=True)


def test_counts_only_record_has_every_field_and_no_timing(run, tmp_path):
    record, _ = run
    path = books.write_record(record, tmp_path)
    saved = json.loads(path.read_text())
    assert saved["status"] == "ok"
    assert saved["provenance"]["sha"] and saved["provenance"]["dirty"] in (True, False)
    assert saved["dataset"]["hash"] == books.dataset_hash(books.SYNTHETIC)
    assert saved["spec_hash"] == books.spec_hash("smoke_poisson")
    assert {"python", "numpy", "scipy", "numba", "tabmat", "blas"} <= set(saved["environment"])
    assert {"cpu_model", "physical_cores", "ram_gib"} <= set(saved["machine"])
    assert saved["threads"] == "pinned"
    for count in ("gram_builds", "pirls_solves", "outer_iterations", "irls_iterations"):
        assert count in saved["work"]
    assert saved["work"]["outer_iterations"] >= 1
    assert {"backend", "convergence", "warnings", "fit"} <= set(saved)
    assert {"objective", "edf"} <= set(saved["fit"])
    assert {"test_deviance", "test_nll", "gini", "ae_worst_dev"} <= set(saved["results"])
    # counts-only drops the clock: no wall, CPU, RSS or phase seconds anywhere
    assert "measurements" not in saved and "phases" not in saved
    assert not [k for k in saved["profile"] if books._is_time(k)]
    index = (tmp_path / "index.jsonl").read_text().splitlines()
    assert len(index) == 1 and json.loads(index[0])["path"] == str(path)


def test_benchboard_payload_follows_worker_protocol_v1(run):
    record, clock = run
    payload = books.fit_payload(record, clock, "task")
    assert payload["protocol"] == 1 and payload["task_id"] == "task"
    (fit,) = payload["fits"]
    assert fit["status"] == "success" and fit["wall_s"] == clock["wall_s"] >= 0
    assert fit["config"]["benchboard_case"] == books.bench_case(record)
    assert fit["summary"]["work"] == record["work"]
    assert fit["summary"]["results"]["gini"] == record["results"]["gini"]
    assert fit["diagnostics"]["key"] == record["key"]
    json.dumps(payload, allow_nan=False)  # benchboard stores strict JSON


def _baseline(record, base_sha):
    base = copy.deepcopy(record)
    base["counts_only"], base["measurements"] = False, {"wall_s": 1.0}
    base["provenance"].update(sha=base_sha, dirty=False)
    base["key"] = dict(base["key"], sha=base_sha)
    return base


def test_compare_pairs_a_record_with_the_baseline_of_the_same_key(run):
    record, _ = run
    base_sha = "a" * 40
    table, refusals = books.compare([record], base_sha, lambda *_: [_baseline(record, base_sha)])
    assert refusals == [] and "| smoke_poisson | pinned |" in table
    # the benchboard case names the whole key but the SHA, so both sides share it
    assert books.bench_case(_baseline(record, base_sha)) == books.bench_case(record)


@pytest.mark.parametrize(
    ("component", "change"),
    [
        ("spec_hash", lambda key: key.update(spec_hash="0" * 64)),
        ("environment.numpy", lambda key: key["environment"].update(numpy="0.0.0")),
        ("machine.cpu_model", lambda key: key["machine"].update(cpu_model="another CPU")),
        ("dataset_hash", lambda key: key.update(dataset_hash="0" * 64)),
        ("harness_hash", lambda key: key.update(harness_hash="0" * 64)),
    ],
)
def test_compare_refuses_a_key_that_differs(run, component, change):
    record, _ = run
    base_sha = "a" * 40
    branch = copy.deepcopy(record)
    change(branch["key"])
    table, refusals = books.compare([branch], base_sha, lambda *_: [_baseline(record, base_sha)])
    assert table is None and len(refusals) == 1
    assert f"{component}: baseline" in refusals[0]
    assert books.bench_case(branch) != books.bench_case(record)


def test_compare_refuses_without_a_usable_baseline(run):
    record, _ = run
    base_sha = "b" * 40
    counts_only = dict(_baseline(record, base_sha), counts_only=True)
    for stored in ([], [counts_only]):
        table, refusals = books.compare([record], base_sha, lambda *_, s=stored: s)
        assert table is None and "no stored baseline" in refusals[0]


def test_nb2_deviance_is_twice_the_log_likelihood_gap_to_the_saturated_model():
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(5)
    y = rng.negative_binomial(2.0, 0.6, 400).astype(float)
    mu = rng.uniform(0.3, 3.0, 400)
    theta = rng.uniform(0.5, 4.0, 400)

    def loglik(mean):
        return stats.nbinom.logpmf(y, theta, theta / (theta + mean)).sum()

    saturated = np.where(y > 0, y, 1e-300)
    expected = 2.0 * (loglik(saturated) - loglik(mu))
    assert books._nb2_deviance(y, mu, theta) == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("reverse", [False, True])
def test_lss_offset_benchmark_fits_declarations_and_prices_new_rows(reverse):
    import numpy as np
    import pandas as pd
    from scipy import stats

    rng = np.random.default_rng(439)
    exposure = rng.uniform(0.3, 1.4, 192)
    mean, size = 3.0 * exposure, 1.2 * np.exp(exposure)
    frame = pd.DataFrame(
        {
            "Exposure": exposure,
            "ClaimNb": rng.negative_binomial(size, size / (size + mean)).astype(float),
        }
    )
    train, test = frame.iloc[:160], frame.iloc[160:]
    spec = copy.deepcopy(books.CASES["lss_nb"])
    spec["predictors"] = [["mean", []], ["theta", []]][:: (-1 if reverse else 1)]
    spec["offsets"] = {"mean": "log_exposure", "theta": "exposure"}
    offsets = books._lss_inputs(spec, train)[3]
    model = books._lss_model(spec, offsets=offsets)
    books._fit(spec, model, train)
    X, y, _, replacements = books._lss_inputs(spec, test)
    parameters = model.predict_parameters(X, offsets=replacements)
    unit = model.predict_parameters(
        test.iloc[:1], offsets={name: np.zeros(1) for name in ("mean", "theta")}
    ).iloc[0]
    np.testing.assert_allclose(parameters["mean"], unit["mean"] * test["Exposure"], rtol=2e-14)
    np.testing.assert_allclose(
        parameters["theta"], unit["theta"] * np.exp(test["Exposure"]), rtol=2e-14
    )
    expected_nll = -stats.nbinom.logpmf(
        y, parameters["theta"], parameters["theta"] / (parameters["theta"] + parameters["mean"])
    ).sum()
    assert books.lss_results(spec, model, test)["test_nll"] == pytest.approx(
        expected_nll, rel=2e-12
    )


def test_offset_exposure_count_metrics_rank_by_rate_and_cut_deciles_by_exposure():
    import numpy as np
    import pandas as pd

    from superglm import families

    rng = np.random.default_rng(9)
    n = 500
    expo = rng.uniform(0.05, 1.0, n)
    rate = rng.uniform(0.05, 0.4, n)
    test = pd.DataFrame({"Exposure": expo, "ClaimNb": rng.poisson(rate * expo).astype(float)})
    spec = dict(books.CASES["fam_nb_fixed_exact"])
    mu = rate * expo
    model = SimpleNamespace(
        predict=lambda X, offset=None: mu,
        _distribution=families.nb2(theta=1.0),
        result=SimpleNamespace(phi=1.0),
    )
    out = books.glm_results(spec, model, test)
    y = test["ClaimNb"].to_numpy()
    assert out["gini"] == pytest.approx(books._gini(y, mu, expo))
    assert out["gini"] != pytest.approx(books._gini(y, mu, np.ones(n)))  # the unfixed count ranking
    assert out["ae_deciles"] == pytest.approx(books._ae(y, mu, books._deciles(rate, expo)))


def test_the_spec_hash_covers_the_harness_metric_code(monkeypatch):
    before = books.spec_hash("smoke_poisson")

    def edited_gini(actual, expected, weights):  # different source, same signature
        return 0.0

    monkeypatch.setattr(books, "_gini", edited_gini)
    assert books.spec_hash("smoke_poisson") != before


def test_a_failed_import_finishes_the_benchboard_task(monkeypatch):
    finished = []

    class History:
        def __init__(self, folder):
            pass

        def start_task(self, task):
            pass

        def import_spools(self, task_id, folder):
            raise OSError("history database unavailable")

        def finish_task(self, task_id, **fields):
            finished.append(fields["status"])

    monkeypatch.delenv("BENCHBOARD_FIT_DIR", raising=False)
    monkeypatch.setattr(books, "_benchboard_history", lambda: type("m", (), {"History": History}))
    monkeypatch.setattr(books, "fit_payload", lambda record, clock, task_id: {})
    record = {"case": "smoke_poisson", "threads": "pinned"}
    with pytest.raises(OSError, match="unavailable"):
        books.record_to_benchboard(record, {"started_at": 0.0, "wall_s": 1.0})
    assert finished == ["failed"]


def test_the_spec_hash_covers_the_work_counter_tables(monkeypatch):
    before = books.spec_hash("smoke_poisson")
    monkeypatch.setattr(books, "PROFILE_WORK", {**books.PROFILE_WORK, "edited": "edited"})
    assert books.spec_hash("smoke_poisson") != before


def test_a_baseline_whose_warm_up_failed_is_not_reused():
    record = {"status": "ok", "counts_only": False, "provenance": {"dirty": False}}
    assert books.usable_baseline({**record, "warmup_error": None})
    assert not books.usable_baseline({**record, "warmup_error": "RuntimeError: compile"})


def test_a_missing_fit_path_package_is_keyed_as_absent(monkeypatch):
    import importlib.metadata

    real = importlib.metadata.version

    def version(name):
        if name == "tabmat":
            raise importlib.metadata.PackageNotFoundError(name)
        return real(name)

    monkeypatch.setattr(importlib.metadata, "version", version)
    assert books.environment()["tabmat"] is None


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "smoke_poisson", "--tree", "../base", "--out", "records"],
        ["run", "smoke_poisson", "--tree=../base", "--out=records"],
    ],
)
def test_a_run_in_another_tree_writes_where_this_checkout_reads(argv, monkeypatch, tmp_path):
    seen = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(books, "_venv_python", lambda tree: tree / "python")
    monkeypatch.setattr(books, "_child", lambda python, tree, args, timeout: seen.append(args) or 0)
    assert books.main(argv) == 0
    (child,) = seen
    assert not any(token.startswith("--tree") for token in child)
    assert child[child.index("--out") + 1] == str(tmp_path / "records")


def test_a_run_keeps_superglms_own_recorder_out_of_the_timed_fit(monkeypatch, tmp_path):
    monkeypatch.setenv("BENCHBOARD_FIT_DIR", str(tmp_path))
    monkeypatch.setenv("BENCHBOARD_TASK_ID", "task-books")
    monkeypatch.delenv("SUPERGLM_FIT_RECORD", raising=False)
    record, _ = books.run_case("smoke_poisson", "pinned", counts_only=True)
    assert record["status"] == "ok"
    assert not list(tmp_path.iterdir())
    assert "SUPERGLM_FIT_RECORD" not in os.environ


def test_the_key_carries_the_observed_pool_widths(run):
    record, _ = run
    observed = record["key"]["threads_observed"]
    assert observed["numba"] == record["threads_observed"]["numba"]
    assert observed["pools"] == record["threads_observed"]["pools"]


def test_baseline_reuse_ignores_the_widths_the_key_step_cannot_see(run, monkeypatch):
    """The key step runs in another process under another wrapper, so its observed widths may
    differ from the timed run's: reuse and the benchboard case name match without them, while
    compare, whose two sides both come from a fit's process, still refuses on them."""
    record, _ = run
    narrowed = copy.deepcopy(record["key"])
    narrowed["threads_observed"] = {"pools": [["openblas", 3]], "numba": 3}
    assert books.reuse_part(narrowed) == books.reuse_part(record["key"])
    assert books.bench_case({"case": "smoke_poisson", "key": narrowed}) == books.bench_case(record)
    stored = copy.deepcopy(record)
    stored.update(key=narrowed, counts_only=False, status="ok", warmup_error=None)
    stored["provenance"] = {**stored["provenance"], "dirty": False}
    _, reason = books.match_baseline(record, [stored], record["key"]["sha"])
    assert reason is not None and "threads_observed" in reason
    assert "--force" in reason  # only the widths differ: the refusal names the remedy
    # ... and the remedy re-times exactly the refused case
    remedy = reason.split("re-time it: books.py ", 1)[1].split()
    seen = []
    monkeypatch.setattr(
        books, "cmd_baseline", lambda args: seen.append(books._case_list(args)) or 0
    )
    assert books.main(remedy) == 0 and seen == [["smoke_poisson"]]


# --------------------------------------------------------------------------- fit-output check


def test_the_record_carries_what_a_user_reads_on_fixed_test_rows(run):
    record, _ = run
    out = record["outputs"]
    _, test = books.load_book(books.SYNTHETIC)
    rows = books._sample_rows(len(test))
    assert out["sample"] == {
        "rows": len(rows),
        "seed": books.SEED,
        "hash": books.digest(rows.tolist()),
    }
    assert len(out["predictions"]) == len(rows) == min(books.OUTPUT_ROWS, len(test))
    assert out["edf_total"] == pytest.approx(sum(out["edf_by_term"].values()) + 1.0)
    assert set(out["smoothing"]) and set(out["quasi_coef_se"]) == set(out["edf_by_term"])
    assert isinstance(out["intercept"], float) and out["summary_se"]["Intercept"] > 0
    assert set(out["decisions"]) == set(books.DECISIONS)
    json.dumps(out, allow_nan=False)  # non-finite entries are null, in the file and in benchboard


def test_the_sample_rows_depend_only_on_the_test_frame_size():
    import numpy as np

    rows = books._sample_rows(50_000)
    assert len(rows) == books.OUTPUT_ROWS == len(np.unique(rows)) and (np.diff(rows) > 0).all()
    np.testing.assert_array_equal(rows, books._sample_rows(50_000))
    np.testing.assert_array_equal(books._sample_rows(10), np.arange(10))


def _check(record, edit=None, base_edit=None, lookup=None):
    base_sha = "c" * 40
    base = _baseline(record, base_sha)
    branch = copy.deepcopy(record)
    if edit:
        edit(branch)
    if base_edit:
        base_edit(base)
    return books.check([branch], base_sha, lookup or (lambda *_: [base]))


def test_a_branch_that_reproduces_its_baseline_passes(run):
    table, failures, refusals = _check(run[0])
    assert failures == [] and refusals == []
    assert "| smoke_poisson | pinned | pass | 0 | 0 | 0 | 0 | 0 | 0 | 0 | same |" in table


def _scale_intercept(rec, factor):
    rec["outputs"]["intercept"] *= factor


def test_a_one_in_a_million_intercept_shift_fails_and_names_the_intercept(run):
    record, _ = run
    table, failures, _ = _check(record, lambda r: _scale_intercept(r, 1 + 1e-6))
    assert len(failures) == 1
    beta0 = record["outputs"]["intercept"]
    expected = abs(beta0) * 1e-6 / max(abs(beta0), 1.0)  # the link-scale floor of 1
    assert f"- FAIL smoke_poisson [pinned] vs {'c' * 12}: intercept moved {expected:.2e}" in table


@pytest.mark.parametrize(
    ("output", "edit"),
    [
        ("predictions", lambda o, f: o["predictions"].__setitem__(7, o["predictions"][7] * f)),
        ("summary_se", lambda o, f: o["summary_se"].update(d=o["summary_se"]["d"] * f)),
        (
            "summary_curve_se",
            lambda o, f: o["summary_curve_se"]["a"].__setitem__(
                1, o["summary_curve_se"]["a"][1] * f
            ),
        ),
        (
            "summary_basis_se",
            lambda o, f: o["summary_basis_se"]["b"].__setitem__(
                2, o["summary_basis_se"]["b"][2] * f
            ),
        ),
        (
            "quasi_coef_se",
            lambda o, f: o["quasi_coef_se"]["a"].__setitem__(0, o["quasi_coef_se"]["a"][0] * f),
        ),
        ("edf_by_term", lambda o, f: o["edf_by_term"].update(a=o["edf_by_term"]["a"] * f)),
        ("edf_total", lambda o, f: o.update(edf_total=o["edf_total"] * f)),
        ("family", lambda o, f: o["family"].update(phi=o["family"]["phi"] * f)),
    ],
)
def test_a_value_fails_just_beyond_the_tolerance_and_passes_just_inside(run, output, edit):
    record, _ = run
    for factor, fails in ((1 + 2 * books.VALUE_RTOL, True), (1 + 0.5 * books.VALUE_RTOL, False)):
        table, failures, _ = _check(record, lambda r, f=factor: edit(r["outputs"], f))
        assert bool(failures) is fails, (output, factor)
        assert (f"{output} moved" in table) is fails
    if output == "predictions":
        table, _, _ = _check(record, lambda r: edit(r["outputs"], 1 + 1e-6))
        assert "predictions moved 1.00e-06 relative at sample row 7" in table


def test_an_entry_near_zero_is_read_against_its_outputs_scale():
    assert books.relative_move([1.0, 1e-9], [1.0, 2e-9], books.FLOOR_FRACTION)[0] == pytest.approx(
        1e-9 / books.FLOOR_FRACTION
    )
    assert books.relative_move([1.0, 1e-9], [1.0, 2e-9], 0.0)[0] == pytest.approx(0.5)
    assert books.relative_move([0.0, 0.0], [0.0, 0.0], 0.0) == (0.0, 0)
    assert books.relative_move([1.0, None], [1.0, 2.0], 0.0) is None  # estimability moved
    assert books.relative_move([1.0], [1.0, 2.0], 0.0) is None


@pytest.mark.parametrize(
    ("decision", "edit"),
    [
        ("converged", lambda r: r["outputs"]["decisions"].update(converged=False)),
        ("termination_reason", lambda r: r["outputs"]["decisions"].update(termination_reason="x")),
        ("backend", lambda r: r["outputs"]["decisions"].update(backend="qr")),
        (
            "warning categories",
            lambda r: r["warnings"].append(
                {"category": "ConvergenceWarning", "message": "m", "count": 1}
            ),
        ),
        ("status", lambda r: r.update(status="raised RuntimeError: x")),
        ("summary_se[d] estimability", lambda r: r["outputs"]["summary_se"].update(d=None)),
        ("edf_by_term entries", lambda r: r["outputs"]["edf_by_term"].pop("a")),
        ("smoothing terms", lambda r: r["outputs"]["smoothing"].pop("a")),
    ],
)
def test_a_decision_must_match_exactly(run, decision, edit):
    table, failures, _ = _check(run[0], edit)
    assert len(failures) == 1
    assert f": {decision} decision differs" in table


def test_a_warning_count_and_a_smoothing_parameter_are_reported_not_gated(run):
    record, _ = run

    def edit(r):
        if r["warnings"]:
            r["warnings"][0] = dict(r["warnings"][0], count=r["warnings"][0]["count"] + 3)
        r["outputs"]["smoothing"]["a"] *= 10.0  # a flat direction's lambda is not identified

    table, failures, _ = _check(record, edit)
    assert failures == [] and "| 2.3e+00 | same |" in table


def test_a_pinned_record_is_checked_only_against_a_pinned_baseline(run):
    record, _ = run
    asked = []
    default = _baseline(record, "c" * 40)
    default["threads"], default["key"] = "default", dict(default["key"], threads="default")

    def lookup(sha):
        asked.append(sha)
        return [default]

    table, failures, refusals = _check(record, lookup=lookup)
    assert asked == ["c" * 40] and table is None and failures == []
    assert "no stored baseline" in refusals[0]
    # a record whose key says another setting is refused even if it were offered
    _, reason = books.match_baseline(record, [default], "c" * 40, usable=books.usable_for_check)
    assert "threads: baseline" in reason


def test_an_untimed_baseline_serves_check_but_not_compare(run):
    record, _ = run
    untimed = dict(_baseline(record, "c" * 40), counts_only=True)
    assert books.usable_for_check(untimed) and not books.usable_baseline(untimed)
    assert not books.usable_for_check(dict(untimed, outputs={"error": "RuntimeError: x"}))
    assert not books.usable_for_check({**untimed, "provenance": {"dirty": True}})
    _, failures, refusals = _check(record, lookup=lambda *_: [untimed])
    assert failures == [] and refusals == []


@pytest.mark.parametrize(("factor", "code"), [(1.0, 0), (1 + 1e-6, 1)])
def test_check_exits_nonzero_when_an_output_moves(run, tmp_path, monkeypatch, factor, code):
    record, _ = run
    base_sha = books._resolve("HEAD")
    branch = copy.deepcopy(record)
    _scale_intercept(branch, factor)
    path = tmp_path / "branch.json"
    path.write_text(json.dumps(branch))
    monkeypatch.setattr(books, "stored_records", lambda *_: [_baseline(record, base_sha)])
    assert books.main(["check", str(path), "--base", "HEAD"]) == code
    monkeypatch.setattr(books, "stored_records", lambda *_: [])
    assert books.main(["check", str(path), "--base", "HEAD"]) == 2  # no baseline: refused


def test_check_reads_a_near_zero_prediction_against_the_predictions_scale(run):
    def tiny(value):
        return lambda r: r["outputs"]["predictions"].__setitem__(0, value)

    _, failures, _ = _check(run[0], tiny(2e-12), base_edit=tiny(1e-12))
    assert failures == []


def test_a_run_from_a_git_hook_still_names_its_commit(monkeypatch, tmp_path):
    """git exports GIT_DIR to its hooks; the pre-push gate runs these tests from one."""
    import superglm

    package = books.Path(superglm.__file__).parent
    expected = books.tree_state(package, package=True)["sha"]
    monkeypatch.setenv("GIT_DIR", str(tmp_path))
    assert expected and books.tree_state(package, package=True)["sha"] == expected


def test_the_recorded_standard_errors_are_the_rendered_summarys(run):
    """Poisson is known-scale: ``summary()`` prints fitted-family SEs (phi = 1), while
    ``metrics().coefficient_se`` is Pearson-scaled; the record keeps both, labelled."""
    record, _ = run
    out = record["outputs"]
    assert set(out["summary_se"]) == {"Intercept", "c[1]", "c[2]", "c[3]", "c[4]", "c[5]", "d"}
    assert set(out["summary_curve_se"]) == {"a", "b"}  # the rows that print a curve SE range
    assert set(out["summary_basis_se"]) == {"a", "b"}
    assert [len(v) for v in out["summary_basis_se"].values()] == [7, 7]
    # the quasi surface is the rendered one times one Pearson factor, which is not 1 here
    ratios = [
        out["quasi_coef_se"]["d"][0] / out["summary_se"]["d"],
        out["quasi_coef_se"]["c"][0] / out["summary_se"]["c[1]"],
        out["quasi_coef_se"]["a"][3] / out["summary_basis_se"]["a"][3],
    ]
    assert abs(ratios[0] - 1.0) > 1e-3 and ratios == pytest.approx([ratios[0]] * 3, rel=1e-12)


def test_check_refuses_a_baseline_that_lacks_an_output_the_branch_records(run):
    record, _ = run
    table, failures, refusals = _check(record, base_edit=lambda b: b["outputs"].pop("summary_se"))
    assert table is None and failures == []
    assert len(refusals) == 1 and "lacks summary_se; re-record it" in refusals[0]

    # a summary that refused on the baseline's side is a decision, not a missing output
    def refuse(b):
        for name in books._SUMMARY_SE_OUTPUTS:
            b["outputs"].pop(name)
        b["outputs"]["coef_se_unavailable"] = "LinAlgError: x"

    table, failures, refusals = _check(record, base_edit=refuse)
    assert refusals == [] and ": summary refusal decision differs" in table


def _fake_children(monkeypatch, tmp_path, record, codes):
    """``check --set``'s children: each case writes a record from ``record`` and exits 0, or
    exits its code in ``codes`` without writing."""
    head = books.tree_state(books.HARNESS_ROOT)["sha"]

    real = books.subprocess.run

    def child(cmd, **kw):
        if cmd[0] != "benchboard":
            return real(cmd, **kw)
        case = cmd[cmd.index("run") + 1]
        if case in codes:
            return SimpleNamespace(returncode=codes[case])
        rec = copy.deepcopy(record)
        rec["case"] = case
        rec["provenance"] = dict(rec["provenance"], sha=head)
        books.write_record(rec, tmp_path)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(books.subprocess, "run", child)
    monkeypatch.setattr(books, "stored_records", lambda *_: [_baseline(record, head)])
    return head


def test_check_set_never_checks_a_record_older_than_its_own_fit(run, tmp_path, monkeypatch, capsys):
    record, _ = run
    head = _fake_children(monkeypatch, tmp_path, record, {"smoke_poisson": 124})
    stale = copy.deepcopy(record)
    stale["provenance"] = dict(stale["provenance"], sha=head)
    books.write_record(stale, tmp_path)  # an earlier, passing record for the same HEAD
    argv = ["check", "--cases", "smoke_poisson", "--base", "HEAD", "--out", str(tmp_path)]
    assert books.main(argv) == 2
    assert (
        "smoke_poisson [pinned]: the branch fit wrote no record (exit 124)"
        in capsys.readouterr().err
    )


def test_check_set_names_a_case_whose_fit_failed_with_no_record(run, tmp_path, monkeypatch, capsys):
    record, _ = run
    _fake_children(monkeypatch, tmp_path, record, {"std_mains": 1})
    argv = ["check", "--cases", "smoke_poisson,std_mains", "--base", "HEAD", "--out", str(tmp_path)]
    assert books.main(argv) == 2
    captured = capsys.readouterr()
    assert "| smoke_poisson | pinned | pass |" in captured.out
    assert "std_mains [pinned]: the branch fit wrote no record (exit 1)" in captured.err
