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
    np.testing.assert_array_equal(partition["row_ids"], [1])
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


def worker_api(name):
    api = admission()
    assert hasattr(api, name), f"Task 3 {name} has not been implemented"
    return getattr(api, name)


@pytest.fixture
def saved_case():
    """Build saved basis state and set fixed coefficients, without response fitting."""
    from superglm import Spline

    x = np.linspace(-2, 3, 31)
    frame = pd.DataFrame({"x": x, "z": np.sin(x * 3), "v": np.cos(x * 2)})
    parents = {name: Spline(kind="cr", k=3) for name in frame}
    for name, parent in parents.items():
        parent.build(frame[name].to_numpy())
    specs = {}
    for right in ("z", "v"):
        spec = TensorInteraction("x", right)
        spec.build(frame["x"].to_numpy(), frame[right].to_numpy(), parents)
        specs[f"x:{right}"] = spec
    beta = np.arange(1, 9, dtype=float) / 4

    def predict(query):
        result = 20 + 2 * query["x"].to_numpy()
        for index, spec in enumerate(specs.values()):
            left, right = spec.parent_names
            result = result + spec.score(
                query[left].to_numpy(), query[right].to_numpy(), beta[index * 4 : index * 4 + 4]
            )
        return result

    selected = SimpleNamespace(
        _interaction_specs=specs,
        _groups=[
            SimpleNamespace(feature_name=name, sl=slice(index * 4, index * 4 + 4))
            for index, name in enumerate(specs)
        ],
        result=SimpleNamespace(beta=beta),
        predict=predict,
    )
    additive = SimpleNamespace(predict=lambda query: np.full(len(query), 50.0))
    partitions = {}
    for name, rows in (("train", np.arange(20)), ("valid", np.arange(20, 31))):
        query = frame.iloc[rows]
        partitions[name] = {
            "features": query,
            "raw": query,
            "response": 20 + 2 * query["x"].to_numpy(),
            "weights": np.ones(len(query)),
            "row_ids": rows,
        }
    models = {"k4_s2": selected, "k4_s0": additive}
    valid = partitions["valid"]
    records = {
        arm: {
            "pairs": [["x", "z"], ["x", "v"]] if arm == "k4_s2" else [],
            "model_pickle_sha256": hashlib.sha256(arm.encode()).hexdigest(),
            "model_pickle_bytes": 100,
            "validation": broad.score(
                valid["response"], model.predict(valid["features"]), "gaussian", valid["weights"]
            ),
            "fit_seconds": 1.0,
            "fit_end_peak_process_rss_mib": 10.0,
            "retained_model_storage": {"total_payload_bytes": 100},
        }
        for arm, model in models.items()
    }
    return {
        "dataset": "uci_airfoil",
        "choice": {
            "chosen_arm": "k4_s2",
            "additive_arm": "k4_s0",
            "matching_additive_arm": "k4_s0",
        },
        "models": models,
        "records": records,
        "partitions": partitions,
        "runtime": {"source_root": str(REPO), "source": {}, "runtime": {}},
        "data": {"data_sha256": "fixture"},
        "preprocessing": {},
        "artifact_root": str(REPO),
    }


def run_saved_fixture(saved_case, tmp_path, monkeypatch):
    worker = worker_api("diagnostic_worker")
    monkeypatch.setattr(admission(), "load_reference_case", lambda *args, **kwargs: saved_case)
    destination = tmp_path / saved_case["dataset"]
    destination.mkdir()
    return worker(saved_case["dataset"], source_root=REPO, output=destination)


def test_worker_exports_every_budget_and_scope_to_aggregate(saved_case, tmp_path, monkeypatch):
    # Catches dropping rank zero, the full budget, a term, or one candidate family.
    import json

    result = run_saved_fixture(saved_case, tmp_path, monkeypatch)
    aggregate = worker_api("aggregate_results")(
        tmp_path, {"uci_airfoil": {"status": "success", "process_seconds": 1.0}}
    )
    attempts = aggregate["cases"]["uci_airfoil"]["attempts"]
    assert len(attempts) == 18
    assert {(a["method"], a["scope"], a["requested_budget"]) for a in attempts} == {
        (method, scope, budget)
        for method in ("rank", "modal")
        for scope in ("term_0", "term_1", "all_terms")
        for budget in (0, 1, 2)
    }
    assert result["status"] == "evaluated"
    for attempt in attempts:
        receipt_path = tmp_path / attempt["receipt_path"]
        assert hashlib.sha256(receipt_path.read_bytes()).hexdigest() == attempt["receipt_sha256"]
        receipt = json.loads(receipt_path.read_text())
        payload = tmp_path / receipt["arrays_path"]
        assert hashlib.sha256(payload.read_bytes()).hexdigest() == receipt["arrays_sha256"]
        assert set(receipt["partitions"]) == {"train", "valid"}
        assert receipt["reference_sha256"] == result["reference_sha256"]
        assert receipt["new_model_fits"] == 0
        assert receipt["fresh_test_evaluation"] is False
        assert receipt["numerical_certificate"] is False
        assert receipt["fitting_speedup_claim"] is False


def test_all_refused_candidates_still_reach_aggregate(saved_case, tmp_path, monkeypatch):
    # A numerical refusal must not turn a requested attempt into a missing row.
    import interaction_compression_geometry as geometry

    def refuse(*args, **kwargs):
        raise ValueError("deliberately singular fixture")

    monkeypatch.setattr(geometry, "product_metric_factors", refuse)
    monkeypatch.setattr(geometry, "saved_penalty_modes", refuse)
    result = run_saved_fixture(saved_case, tmp_path, monkeypatch)
    aggregate = worker_api("aggregate_results")(
        tmp_path, {"uci_airfoil": {"status": "success", "process_seconds": 1.0}}
    )
    attempts = aggregate["cases"]["uci_airfoil"]["attempts"]
    assert result["status"] == "no_admissible_candidate"
    assert len(attempts) == 18
    assert all(a["status"] == "refused" and a["refusal_reason"] for a in attempts)


def test_modal_refusal_preserves_rank_evidence(saved_case, tmp_path, monkeypatch):
    import interaction_compression_geometry as geometry

    def refuse(*args, **kwargs):
        raise ValueError("unresolved modal fixture")

    monkeypatch.setattr(geometry, "saved_penalty_modes", refuse)
    result = run_saved_fixture(saved_case, tmp_path, monkeypatch)
    assert sum(a["status"] == "evaluated" for a in result["attempts"]) == 9
    assert all(a["status"] == "refused" for a in result["attempts"] if a["method"] == "modal")


def test_diagnostic_worker_cannot_fit_models(saved_case, tmp_path, monkeypatch):
    from superglm import SuperGLM

    def forbidden_fit(*args, **kwargs):
        pytest.fail("The no-fit diagnostic called a model fitting method")

    monkeypatch.setattr(SuperGLM, "fit", forbidden_fit)
    monkeypatch.setattr(SuperGLM, "fit_reml", forbidden_fit)
    result = run_saved_fixture(saved_case, tmp_path, monkeypatch)
    assert result["status"] == "evaluated"
    assert all(a["new_model_fits"] == 0 for a in result["attempts"])


def test_rank_zero_keeps_joint_mains_and_accounts_storage(saved_case, tmp_path, monkeypatch):
    import json

    result = run_saved_fixture(saved_case, tmp_path, monkeypatch)
    for summary in result["attempts"]:
        if summary["method"] != "rank" or summary["scope"] != "all_terms":
            continue
        receipt = json.loads((tmp_path / summary["receipt_path"]).read_text())
        budget = receipt["requested_budget"]
        storage = receipt["storage"]
        assert storage["original_coefficient_entries"] == 8
        assert storage["logical_factor_entries"] == 8 * budget
        assert storage["logical_candidate_payload_bytes"] == 64 * budget
        assert storage["expanded_candidate_coefficient_bytes"] == 64
        assert storage["diagnostic_array_payload_bytes"] > 64
        if budget == 0:
            with np.load(tmp_path / receipt["arrays_path"]) as arrays:
                for name, partition in saved_case["partitions"].items():
                    expected = 20 + 2 * partition["features"]["x"].to_numpy()
                    allowance = arrays[f"{name}__predictor_update_allowance"]
                    assert np.all(np.abs(arrays[f"{name}__prediction"] - expected) <= allowance)
                    assert not np.array_equal(
                        arrays[f"{name}__prediction"], np.full(len(expected), 50)
                    )
                    assert (
                        receipt["partitions"][name]["additive_controls"]["matching_additive_arm"][
                            "loss"
                        ]
                        > 0
                    )
            assert receipt["rank_zero_is_additive_refit"] is False


def test_full_budgets_replay_predictor_and_loss_with_derived_bounds(
    saved_case, tmp_path, monkeypatch
):
    import json

    result = run_saved_fixture(saved_case, tmp_path, monkeypatch)
    full = [a for a in result["attempts"] if a["requested_budget"] == 2]
    assert len(full) == 6
    for summary in full:
        receipt = json.loads((tmp_path / summary["receipt_path"]).read_text())
        assert receipt["full_budget"] is True
        with np.load(tmp_path / receipt["arrays_path"]) as arrays:
            for name, metrics in receipt["partitions"].items():
                reference = saved_case["models"]["k4_s2"].predict(
                    saved_case["partitions"][name]["features"]
                )
                error = np.abs(arrays[f"{name}__prediction"] - reference)
                assert np.all(error <= arrays[f"{name}__full_budget_predictor_allowance"])
                assert (
                    abs(metrics["loss"] - metrics["reference_loss"])
                    <= metrics["full_budget_loss_allowance"]
                )
                assert metrics["full_budget_replay_passed"] is True


def test_large_base_updates_include_rounded_addition_in_allowance():
    # An omitted addition term gives a tiny contraction bound despite a 1-ulp base update.
    update = worker_api("replace_interactions")
    result = update(
        np.array([1e16]),
        [
            {
                "left": np.ones((1, 1)),
                "right": np.ones((1, 1)),
                "reference_coefficients": np.ones((1, 1)),
                "candidate_coefficients": np.zeros((1, 1)),
                "reference_effect": np.ones(1),
                "runtime_discrepancy": np.zeros(1),
                "coefficient_allowance": 0.0,
            }
        ],
    )
    assert result["prediction"][0] == 1e16
    assert result["predictor_update_allowance"][0] >= 1.0


@pytest.mark.parametrize("additive,allowed", [(1.0, False), (1.0 + 1e-15, False), (2.0, True)])
def test_gain_denominator_must_exceed_its_derived_band(additive, allowed):
    gain = worker_api("gain_retention")(
        additive_loss=additive,
        reference_loss=1.0,
        candidate_loss=0.5,
        additive_allowance=2e-15,
        reference_allowance=2e-15,
        candidate_allowance=3e-15,
    )
    assert (gain["gain_retention"] is not None) is allowed
    if allowed:
        assert gain["gain_retention"] == 1.5  # Raw, deliberately greater than one.
        assert gain["gain_retention_allowance"] > 0
    else:
        assert gain["gain_ratio_status"] == "denominator_not_positive_beyond_allowance"


def test_existing_batch_output_is_refused_before_launch(tmp_path):
    run = worker_api("run_batch")
    with pytest.raises(FileExistsError):
        run(manifest_path=admission().MANIFEST, source_root=REPO, output=tmp_path)


def test_nonfinite_loss_bound_refuses_instead_of_admitting_an_infinite_band():
    api = admission()
    update = {
        "prediction": np.array([1e154]),
        "predictor_update_allowance": np.array([1e154]),
        "coefficient_effect_allowance": np.zeros(1),
        "full_budget_predictor_allowance": np.array([1e154]),
    }
    with pytest.raises(ValueError, match="allowance.*finite|scale"):
        api._partition_metrics(
            {"response": np.zeros(1), "weights": np.ones(1)}, np.zeros(1), {}, update, False
        )


def test_timeout_aggregate_preserves_completed_attempts(saved_case, tmp_path, monkeypatch):
    result = run_saved_fixture(saved_case, tmp_path, monkeypatch)
    # The owner timed out after case output was written. It must remain a failed process.
    aggregate = worker_api("aggregate_results")(
        tmp_path,
        {
            "uci_airfoil": {"status": "timeout", "process_seconds": 180.0},
            "uci_concrete": {"status": "error", "returncode": 1, "process_seconds": 1.0},
        },
    )
    assert aggregate["status"] == "incomplete"
    assert aggregate["attempt_count"] == len(result["attempts"]) == 18
    assert aggregate["cases"]["uci_airfoil"]["process"]["status"] == "timeout"
    assert aggregate["cases"]["uci_concrete"]["attempts"] == []


def test_modal_logical_storage_includes_needed_mode_columns(saved_case, tmp_path, monkeypatch):
    result = run_saved_fixture(saved_case, tmp_path, monkeypatch)
    attempts = [
        a for a in result["attempts"] if a["method"] == "modal" and a["scope"] == "all_terms"
    ]
    assert [a["storage"]["logical_selected_product_entries"] for a in attempts] == [0, 2, 8]
    assert [a["storage"]["logical_mode_column_entries"] for a in attempts] == [0, 8, 16]
    assert [a["storage"]["logical_candidate_payload_bytes"] for a in attempts] == [0, 80, 192]
    assert all(a["storage"]["retained_frozen_modal_basis_payload_bytes"] == 128 for a in attempts)


def test_aggregate_tracks_product_metric_refusals_separately(saved_case, tmp_path, monkeypatch):
    import interaction_compression_geometry as geometry

    def refuse(*args, **kwargs):
        raise ValueError("metric unavailable")

    monkeypatch.setattr(geometry, "product_metric_factors", refuse)
    result = run_saved_fixture(saved_case, tmp_path, monkeypatch)
    aggregate = worker_api("aggregate_results")(
        tmp_path, {"uci_airfoil": {"status": "success", "process_seconds": 1.0}}
    )
    assert len(result["attempts"]) == 18
    assert aggregate["refused_count"] == 9
    assert aggregate["evaluated_count"] == 9
    assert aggregate["attempts_with_product_metric_refusal"] == 9


def test_interrupted_worker_preserves_completed_attempts(saved_case, tmp_path, monkeypatch):
    api = admission()
    original = api._term_candidate
    calls = 0

    def interrupt_after_two(*args):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt
        return original(*args)

    monkeypatch.setattr(api, "_term_candidate", interrupt_after_two)
    with pytest.raises(KeyboardInterrupt):
        run_saved_fixture(saved_case, tmp_path, monkeypatch)
    aggregate = api.aggregate_results(
        tmp_path, {"uci_airfoil": {"status": "timeout", "process_seconds": 180.0}}
    )
    assert aggregate["status"] == "incomplete"
    assert aggregate["evaluated_count"] == 2
    assert aggregate["cases"]["uci_airfoil"]["unrun_attempt_count"] == 16
    figures = api.plot_results(aggregate, tmp_path)
    assert len(figures) == 1
    assert (tmp_path / figures[0]["path"]).is_file()


def test_batch_deadline_interrupts_and_reaps_owned_worker(tmp_path):
    import os
    import signal
    import sys

    from benchmark_housing_tensor import run_isolated

    deadline = worker_api("batch_deadline")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    pid_file = tmp_path / "pid"
    command = [
        sys.executable,
        "-c",
        "import os,signal,sys,time; from pathlib import Path; "
        "Path(sys.argv[1]).write_text(str(os.getpid())); "
        "os.kill(os.getppid(),signal.SIGALRM); time.sleep(30)",
        str(pid_file),
    ]
    with pytest.raises(TimeoutError, match="batch"):
        with deadline():
            run_isolated(
                command, log_path=tmp_path / "worker.log", timeout=180, env=os.environ.copy()
            )
    assert signal.getsignal(signal.SIGALRM) is previous_handler
    assert signal.getitimer(signal.ITIMER_REAL) == previous_timer
    pid = int(pid_file.read_text())
    with pytest.raises(ChildProcessError):
        os.waitpid(pid, os.WNOHANG)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
