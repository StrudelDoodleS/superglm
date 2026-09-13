"""Keep joins and shared-seed uncertainty valid for the booster comparison."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import psst_booster_analysis as analysis
import pytest
from psst_booster_analysis import clustered_difference, join_boosters


def test_join_does_not_pair_by_completion_order_or_drop_failed_jobs():
    reference = {"a": {"models": {}}, "b": {"models": {}}}
    records = [
        {"backend": "xgboost", "id": "b", "status": "failed"},
        {"backend": "xgboost", "id": "a", "status": "ok", "test_loss": 2.0, "test_risk": 1.0},
    ]
    joined = join_boosters(reference, records, backends=("xgboost",))
    assert joined["a"]["models"]["xgboost"]["test_risk"] == 1.0
    assert joined["b"]["models"]["xgboost"]["status"] == "failed"
    with pytest.raises(ValueError, match="identit"):
        join_boosters(reference, records[:1], backends=("xgboost",))
    with pytest.raises(ValueError, match="duplicate"):
        join_boosters(reference, [*records, records[0]], backends=("xgboost",))


def test_repeating_shared_seed_cells_does_not_shrink_uncertainty():
    rows = [
        {
            "design": "gaussian_independent",
            "replicate": i,
            "models": {"psst": {"test_risk": 4.0}, "xgboost": {"test_risk": value}},
        }
        for i, value in enumerate((1.0, 2.0, 5.0, 8.0))
    ]
    once = clustered_difference(rows, "xgboost", "test_risk")
    repeated = clustered_difference([row for row in rows for _ in range(5)], "xgboost", "test_risk")
    assert once["difference"] == repeated["difference"] == 0.0
    np.testing.assert_array_equal(once["difference_ci"], repeated["difference_ci"])
    assert once["difference_ci"][0] < 0 < once["difference_ci"][1]


def test_shared_booster_seed_is_clustered_across_designs():
    rows = [
        {
            "design": design,
            "replicate": replicate,
            "models": {"psst": {"test_risk": 2.0}, "xgboost": {"test_risk": value}},
        }
        for design, values in (("independent", (1.0, 3.0)), ("correlated", (3.0, 1.0)))
        for replicate, value in enumerate(values)
    ]
    # Both design contributions cancel within each shared random-seed block.
    result = clustered_difference(rows, "xgboost", "test_risk")
    np.testing.assert_array_equal(result["difference_ci"], [0.0, 0.0])


@pytest.fixture
def reference_context():
    receipt = json.loads(Path(__file__).with_name("psst_fast_receipt.json").read_text())
    return receipt["manifests"], receipt["fast_metadata"]


def test_matching_reference_provenance_accepts_different_output_directories(reference_context):
    manifests, metadata = reference_context
    analysis.validate_fast_reference(manifests["psst"], manifests["fast"], metadata)


@pytest.mark.parametrize("field", ["package_source_sha256", "script_sha256", "numpy", "arguments"])
def test_different_fast_manifest_is_refused(reference_context, field):
    manifests, metadata = copy.deepcopy(reference_context)
    if field == "arguments":
        manifests["fast"][field]["refit_replicates"] += 1
    else:
        manifests["fast"][field] = "different execution"
    with pytest.raises(ValueError, match="FAST reference manifest"):
        analysis.validate_fast_reference(manifests["psst"], manifests["fast"], metadata)


@pytest.mark.parametrize(
    "field",
    ["wrapper_sha256", "protocol_sha256", "methods", "native_purify_flag", "interpret_core"],
)
def test_different_fast_wrapper_metadata_is_refused(reference_context, field):
    manifests, metadata = copy.deepcopy(reference_context)
    metadata[field] = "different wrapper"
    with pytest.raises(ValueError, match="FAST wrapper metadata"):
        analysis.validate_fast_reference(manifests["psst"], manifests["fast"], metadata)


def test_main_rejects_fast_provenance_before_joining_rows(reference_context, monkeypatch, tmp_path):
    manifests, metadata = copy.deepcopy(reference_context)
    inputs = tmp_path / "boosters"
    inputs.mkdir()
    manifest = {
        "tasks": 1800,
        "source_hashes": {"psst_detection_study.py": manifests["psst"]["script_sha256"]},
        "package_source_sha256": manifests["psst"]["package_source_sha256"],
        "numpy": manifests["psst"]["numpy"],
    }
    (inputs / "study-manifest.json").write_text(json.dumps(manifest))
    (inputs / "study.jsonl").write_text("{}\n" * 1800)
    manifests["fast"]["package_source_sha256"] = "another implementation"
    for name, directory in (("psst", "final"), ("fast", "fast-final")):
        path = tmp_path / directory
        path.mkdir()
        (path / "study-manifest.json").write_text(json.dumps(manifests[name]))
    (tmp_path / "fast-final/fast-metadata.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(
        sys, "argv", ["analysis", "--input", str(inputs), "--reference", str(tmp_path)]
    )

    def unexpected_join(*paths):
        raise AssertionError("Reference rows must not be joined with different FAST provenance")

    monkeypatch.setattr(analysis, "read_references", unexpected_join)
    with pytest.raises(ValueError, match="FAST reference manifest"):
        analysis.main()
