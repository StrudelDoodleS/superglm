"""Admission keeps saved artifacts pinned and prediction access developmental."""

import copy
import hashlib
import importlib
import pickle
from pathlib import Path
from types import SimpleNamespace

import benchmark_broad_interactions as broad
import broad_interaction_data as data
import numpy as np
import pandas as pd
import pytest

from superglm import Gaussian, IdentityLink, LogLink, Poisson, TensorInteraction

REPO = Path(__file__).resolve().parents[1]


def admission():
    assert importlib.util.find_spec("benchmark_interaction_compressibility") is not None, (
        "The saved-model admission helpers have not been implemented"
    )
    return importlib.import_module("benchmark_interaction_compressibility")


def test_fixed_manifest_admits_only_the_approved_development_cases():
    api = admission()
    manifest = api.read_manifest()
    measurement = api.read_measurement(manifest, REPO)
    assert manifest["cases"] == {
        "uci_airfoil": "k4_s2",
        "uci_concrete": "k6_s4",
        "kaggle_king_county_sales": "k6_s4",
    }
    assert manifest["partitions"] == ["train", "valid"]
    assert manifest["fit_allowed"] is manifest["test_evaluation_allowed"] is False
    assert measurement["datasets"]["uci_airfoil"]["choice"]["chosen_arm"] == "k4_s2"


def test_changed_measurement_bytes_are_refused(tmp_path):
    api = admission()
    manifest = api.read_manifest()
    destination = tmp_path / manifest["measurement_path"]
    destination.parent.mkdir(parents=True)
    destination.write_text("{}")
    with pytest.raises(ValueError, match="measurement.*hash"):
        api.read_measurement(manifest, tmp_path)


def test_runtime_checks_the_imported_package_and_frozen_source_before_pickle(monkeypatch):
    api = admission()
    measurement = {"protocol": {"source": broad.source_identity()}, "runtime": broad.gbm.runtime()}
    admitted = api.admit_runtime(REPO, measurement)
    assert admitted["source"] == measurement["protocol"]["source"]
    changed = copy.deepcopy(measurement)
    changed["protocol"]["source"]["existing"]["package_source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source identity"):
        api.admit_runtime(REPO, changed)
    import superglm

    monkeypatch.setattr(superglm, "__file__", "/wrong/src/superglm/__init__.py")
    with pytest.raises(ValueError, match="imported.*source"):
        api.admit_runtime(REPO, measurement)


@pytest.mark.parametrize("fault", ["version", "threads"])
def test_changed_runtime_versions_or_threadpools_are_refused(fault):
    api = admission()
    measurement = {"protocol": {"source": broad.source_identity()}, "runtime": broad.gbm.runtime()}
    changed = copy.deepcopy(measurement)
    if fault == "version":
        changed["runtime"]["packages"]["numpy"] = "0.0.0"
    else:
        changed["runtime"]["threadpools"][0]["num_threads"] = 2
    with pytest.raises(ValueError, match="runtime"):
        api.admit_runtime(REPO, changed)


@pytest.mark.parametrize("fault", ["family", "link", "term_class", "decomposed", "pair_order"])
def test_model_admission_refuses_unvalidated_families_links_and_terms(fault):
    api = admission()
    spec = TensorInteraction("x", "z")
    spec._p1, spec._p2 = 1, 2
    model = SimpleNamespace(
        _distribution=Gaussian(),
        _link=IdentityLink(),
        _interaction_specs={"x:z": spec},
        _groups=[SimpleNamespace(feature_name="x:z", sl=slice(0, 2))],
        result=SimpleNamespace(beta=np.array([1.0, 2.0])),
    )
    if fault == "family":
        model._distribution = Poisson()
    elif fault == "link":
        model._link = LogLink()
    elif fault == "term_class":
        model._interaction_specs["x:z"] = SimpleNamespace(**vars(spec))
    elif fault == "decomposed":
        spec._decompose = True
    pairs = [["z", "x"]] if fault == "pair_order" else [["x", "z"]]
    with pytest.raises((TypeError, ValueError)):
        api.admit_model(model, pairs)


def test_development_partition_never_accesses_test_rows_or_outcomes():
    api = admission()

    class DevelopmentRows(dict):
        def __getitem__(self, name):
            assert name in ("train", "valid"), "Forbidden test partition access"
            return super().__getitem__(name)

    prepared = {
        "frame": pd.DataFrame({"x": [2.0, 4.0, 8.0], "y": [3.0, 5.0, np.nan]}),
        "entry": {"id": "uci_airfoil", "features": ["x"], "primary_target": "y"},
        "rows": DevelopmentRows(train=np.array([0]), valid=np.array([1])),
        "sample_weight": np.ones(3),
        "state": {
            "input_columns": ["x"],
            "features": {"x": {"kind": "spline", "center": 2.0, "scale": 2.0, "median": 2.0}},
        },
    }
    partition = api.development_partition(prepared, "valid")
    np.testing.assert_array_equal(partition["raw"]["x"], [4.0])
    np.testing.assert_array_equal(partition["features"]["x"], [1.0])
    np.testing.assert_array_equal(partition["response"], [5.0])
    with pytest.raises(ValueError, match="train.*valid"):
        api.development_partition(prepared, "test")


@pytest.mark.parametrize("fault", ["source", "data", "model_hash", "unselected"])
def test_case_admission_refuses_before_unpickling(tmp_path, monkeypatch, fault):
    api = admission()
    payload = pickle.dumps({"would_be_loaded": True})
    model_hash = hashlib.sha256(payload).hexdigest()
    metadata = {"family": "gaussian", "source_bytes_verified": True, "data_sha256": "abc"}
    measurement = {
        "protocol": {"source": broad.source_identity()},
        "runtime": broad.gbm.runtime(),
        "datasets": {
            "uci_airfoil": {
                "data": metadata,
                "choice": {
                    "chosen_arm": "k4_s2",
                    "additive_arm": "k6_s0",
                    "matching_additive_arm": "k4_s0",
                },
                "fits": {
                    arm: {
                        "status": "converged",
                        "pairs": [],
                        "validation": {
                            "primary_loss": 1.0,
                            "mse": 1.0,
                            "rows": 1,
                            "weight_sum": 1.0,
                        },
                        "model_pickle_sha256": model_hash,
                        "model_pickle_bytes": len(payload),
                    }
                    for arm in ("k4_s2", "k6_s0", "k4_s0")
                },
            }
        },
    }
    manifest = api.read_manifest()
    for arm in ("k4_s2", "k6_s0", "k4_s0"):
        destination = tmp_path / manifest["run_root"] / "uci_airfoil" / arm / "model.pkl"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(payload)
    monkeypatch.setattr(api, "read_measurement", lambda *_: measurement)
    monkeypatch.setattr(data, "load_prepared", lambda *args, **kwargs: {"metadata": metadata})

    def forbidden_load(*args, **kwargs):
        pytest.fail("A refusal happened too late: pickle was opened")

    monkeypatch.setattr(pickle, "loads", forbidden_load)
    monkeypatch.setattr(pickle, "load", forbidden_load)
    if fault == "source":
        measurement["protocol"]["source"]["existing"]["package_source_sha256"] = "0" * 64
    elif fault == "data":
        measurement["datasets"]["uci_airfoil"]["data"] = {**metadata, "data_sha256": "changed"}
    elif fault == "model_hash":
        measurement["datasets"]["uci_airfoil"]["fits"]["k4_s0"]["model_pickle_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        api.load_reference_case(
            "uci_abalone" if fault == "unselected" else "uci_airfoil",
            source_root=REPO,
            artifact_root=tmp_path,
        )


def test_saved_validation_replay_uses_the_frozen_score_and_refuses_changed_loss():
    api = admission()
    partition = {
        "features": pd.DataFrame({"x": [2.0, 5.0]}),
        "response": np.array([3.0, 1.0]),
        "weights": np.ones(2),
    }
    model = SimpleNamespace(predict=lambda frame: frame["x"].to_numpy())
    recorded = {"primary_loss": 8.5, "mse": 8.5, "rows": 2, "weight_sum": 2.0}
    receipt = api.replay_validation(model, partition, recorded)
    assert receipt["absolute_loss_difference"] == 0
    assert receipt["replayed"] == recorded
    with pytest.raises(ValueError, match="validation.*replay"):
        api.replay_validation(model, partition, {**recorded, "primary_loss": 9.0})
