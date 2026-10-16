"""benchmarks/books.py: a counts-only record carries every field, and compare refuses a mismatched key.

The harness's numbers stand in for re-timing master in pull request bodies, so
its records must say what was measured and it must refuse a baseline whose key
(dataset, spec, environment, machine, thread setting) differs from the branch's.
"""

from __future__ import annotations

import copy
import json

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
    assert {"python", "numpy", "scipy", "numba", "blas"} <= set(saved["environment"])
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
