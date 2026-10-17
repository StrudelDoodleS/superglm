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
