"""Replay a complete synthetic audit with interrupted fit workers and no model fits."""

import importlib.util
import json
import pickle
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

DATASET = "uci_airfoil"
FAILED_ARM = "k4_s1"
WORKER_KEY = f"{DATASET}/{FAILED_ARM}"
BROKEN_JSON = {"empty": b"", "truncated": b'{"status": "fitting"'}
IDENTITIES = ("source", "protocol_sha256", "data", "data_identity_sha256", "proposal_result_sha256")


@pytest.fixture
def audit_run(tmp_path, monkeypatch):
    path = (
        Path(__file__).resolve().parents[1]
        / "notes/research/check_broad_interaction_measurements.py"
    )
    spec = importlib.util.spec_from_file_location("partial_receipt_audit", path)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    broad, base, gbm = audit.broad, audit.base, audit.gbm
    entry = {
        "id": DATASET,
        "features": ["x", "z"],
        "primary_target": "y",
        "split": {"strategy": "group", "columns": ["group"], "seed": 19},
        "data_kind": "synthetic audit fixture",
        "counts_as_real_source": False,
    }
    frame = pd.DataFrame({"x": np.arange(30) % 4, "z": np.arange(30) % 3, "group": np.arange(30)})
    frame["y"] = frame["x"] + frame["z"]
    prepared = audit.data.prepare_frame(frame, entry)
    source = {"audit_fixture_sha256": gbm.file_hash(Path(__file__))}
    monkeypatch.setattr(broad, "source_identity", lambda: source)
    monkeypatch.setattr(audit.data, "load_prepared", lambda *args, **kwargs: prepared)

    def build(
        raw_kind="partial", *, process_status="timeout", missing=(), changed=None, leftover=False
    ):
        root = tmp_path / "run"
        args = SimpleNamespace(
            output=root,
            data_root=tmp_path,
            datasets=[DATASET],
            fit_timeout=5.0,
            case_budget=30.0,
            total_budget=60.0,
        )

        def isolated(command, *, log_path, timeout, env):
            stage = command[command.index("--stage") + 1]
            arm = command[command.index("--arm") + 1]
            folder = Path(command[command.index("--output") + 1])
            case_root = folder.parent
            log_path.write_text("Synthetic child receipt; no fit was run.\n")
            failed = stage == "fit" and arm == FAILED_ARM
            cost = 5.0 if failed else {"propose": 1.0, "fit": 2.0, "evaluate": 3.0}[stage]
            process = {
                "status": process_status if failed else "success",
                "pid": 123,
                "returncode": -9 if failed and process_status == "timeout" else int(failed),
                "process_seconds": cost,
            }
            record = {
                "dataset": DATASET,
                "arm": arm,
                "stage": stage,
                "started_utc": datetime.now(UTC).isoformat(),
                "source": source,
                "protocol_sha256": gbm.file_hash(root / "protocol.json"),
                "data": prepared["metadata"],
                "data_identity_sha256": base.digest_json(prepared["metadata"]),
                "runtime": {
                    "python": "synthetic child",
                    "platform": "synthetic child",
                    "packages": {},
                    "threadpools": [{"num_threads": 1}],
                },
                "rows": {name: len(rows) for name, rows in prepared["rows"].items()},
                "raw_predictor_count": 2,
            }
            filename = "evaluation.json" if stage == "evaluate" else "result.json"
            if stage == "propose":
                record.update(
                    status="proposed",
                    admission=broad.admit_pairs(prepared["state"], [["x", "z"]]),
                    proposal={
                        "pairs": [["x", "z"]],
                        "scores": [{"pair": ["x", "z"], "score": 1.0}],
                        "policy": "synthetic proposal",
                        "scope": "training rows only",
                        "model_parameters": {},
                        "timing": {},
                        "tree_diagnostics": {},
                        "retained_model_storage": {},
                        "fit_end_peak_process_rss_mib": 1.0,
                    },
                )
            elif stage == "fit":
                pairs = [["x", "z"]] if failed else []
                record.update(
                    status="fitting",
                    parent_k=4,
                    pairs=pairs,
                    nominal=broad.nominal_size(prepared["state"], 4, pairs),
                    proposal_result_sha256=gbm.file_hash(case_root / "proposals/result.json"),
                )
                if not failed or leftover:
                    # The audit only hashes this valid pickle; it must never load it.
                    model_path = folder / "model.pkl"
                    model_path.write_bytes(pickle.dumps(None))
                    record.update(
                        validation={
                            "primary_loss": 0.0 if failed else 1.0,
                            "mse": 0.0 if failed else 1.0,
                            "rows": len(prepared["rows"]["valid"]),
                            "weight_sum": float(len(prepared["rows"]["valid"])),
                        },
                        model_pickle_sha256=gbm.file_hash(model_path),
                        model_pickle_bytes=model_path.stat().st_size,
                    )
                if failed:
                    if raw_kind in BROKEN_JSON:
                        (folder / filename).write_bytes(BROKEN_JSON[raw_kind])
                    elif raw_kind != "none":
                        for key in missing:
                            record.pop(key)
                        record.update(changed or {})
                        record["warnings"] = []
                        base.write_json(folder / filename, record)
                    return process
                record.update(
                    status="converged",
                    P=2,
                    q=0,
                    convergence={"combined_converged": True},
                    fit_seconds=1.0,
                )
            else:
                rows = prepared["rows"]["test"]
                response = frame.iloc[rows]["y"].to_numpy(dtype=float)
                prediction_path = folder / "test_predictions.npz"
                np.savez_compressed(
                    prediction_path,
                    row_index=rows,
                    response=response,
                    prediction=response + 1.0,
                    sample_weight=prepared["sample_weight"][rows],
                )
                record.update(
                    status="evaluated",
                    choice_sha256=gbm.file_hash(case_root / "choice.json"),
                    test={
                        "primary_loss": 1.0,
                        "mse": 1.0,
                        "rows": len(rows),
                        "weight_sum": float(len(rows)),
                    },
                    test_predictions_sha256=gbm.file_hash(prediction_path),
                    test_prediction_seconds=0.1,
                )
            record.update(
                warnings=[],
                finished_utc=datetime.now(UTC).isoformat(),
                worker_end_peak_process_rss_mib=1.0,
            )
            base.write_json(folder / filename, record)
            return process

        monkeypatch.setattr(broad, "run_isolated", isolated)
        broad.run_suite(args)
        suite = audit.read_json(root / "suite.json")
        return SimpleNamespace(audit=audit, root=root, suite=suite, failed=root / WORKER_KEY)

    return build


@pytest.mark.parametrize("raw_kind", ["none", "empty", "truncated", "partial"])
@pytest.mark.parametrize("process_status", ["error", "timeout"])
def test_audit_preserves_interrupted_attempt_costs_and_evidence(
    audit_run, raw_kind, process_status
):
    run = audit_run(raw_kind, process_status=process_status)
    summary = run.audit.summarize(run.root)
    case = summary["datasets"][DATASET]
    failed = case["fits"][FAILED_ARM]
    assert summary["totals"]["fit_statuses"] == {"converged": 1, process_status: 1}
    assert summary["totals"]["search_worker_process_seconds"] == 8.0
    assert summary["totals"]["evaluation_worker_process_seconds"] == 3.0
    assert summary["totals"]["all_worker_process_seconds"] == 11.0
    assert case["all_worker_process_seconds"] == 11.0
    assert case["outcome"] == "additive_retained"
    assert case["choice"]["evaluation_arms"] == ["k4_s0"]
    assert failed["process_seconds"] == 5.0
    assert failed["warnings_complete"] is False
    assert (
        failed["parent_finished_utc"]
        == run.suite["datasets"][DATASET]["arms"][FAILED_ARM]["parent_finished_utc"]
    )
    checks = summary["audit"]
    assert checks["all_declared_arms_accounted_for"] == 2
    assert checks["test_scores_exactly_replayed_from_saved_arrays"] == 1
    complete = raw_kind == "partial"
    assert checks["all_data_and_split_identities_match_repreparation"] is complete
    assert checks["all_worker_threadpools_one_thread"] is complete
    for key in ("failed_workers_with_incomplete_identity", "workers_with_unverified_threadpools"):
        if complete:
            assert key not in checks
        else:
            assert checks[key] == [WORKER_KEY]
    if raw_kind in BROKEN_JSON:
        receipt = failed["incomplete_receipt"]
        preserved = run.failed / receipt["path"]
        assert preserved.read_bytes() == BROKEN_JSON[raw_kind]
        assert receipt["bytes"] == len(BROKEN_JSON[raw_kind])
        assert (
            summary["raw_artifact_sha256"][str(preserved.relative_to(run.root))]
            == receipt["sha256"]
        )
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("missing", [*IDENTITIES, "runtime"])
def test_audit_reports_missing_identity_and_runtime_independently(audit_run, missing):
    run = audit_run(missing=(missing,))
    checks = run.audit.summarize(run.root)["audit"]
    assert checks["all_data_and_split_identities_match_repreparation"] is (missing == "runtime")
    assert checks["all_worker_threadpools_one_thread"] is (missing != "runtime")
    expected = (
        "workers_with_unverified_threadpools"
        if missing == "runtime"
        else "failed_workers_with_incomplete_identity"
    )
    assert checks[expected] == [WORKER_KEY]


@pytest.mark.parametrize("field", IDENTITIES)
def test_audit_rejects_known_bad_identity_even_with_other_evidence_missing(audit_run, field):
    other = "data_identity_sha256" if field == "source" else "source"
    run = audit_run(missing=(other, "runtime"), changed={field: "wrong identity"})
    with pytest.raises(AssertionError):
        run.audit.summarize(run.root)


@pytest.mark.parametrize("raw_kind", ["empty", "truncated"])
def test_audit_rejects_tampering_with_preserved_incomplete_bytes(audit_run, raw_kind):
    run = audit_run(raw_kind)
    record = run.suite["datasets"][DATASET]["arms"][FAILED_ARM]
    preserved = run.failed / record["incomplete_receipt"]["path"]
    preserved.write_bytes(preserved.read_bytes() + b" ")
    with pytest.raises(AssertionError):
        run.audit.summarize(run.root)


@pytest.mark.parametrize("process_status", ["error", "timeout"])
def test_interrupted_serialization_artifacts_remain_excluded_from_selection(
    audit_run, process_status
):
    run = audit_run(process_status=process_status, leftover=True)
    summary = run.audit.summarize(run.root)
    case = summary["datasets"][DATASET]
    failed = case["fits"][FAILED_ARM]
    assert failed["validation"]["primary_loss"] == 0.0
    assert failed["selection_eligible"] is False
    assert case["choice"]["chosen_arm"] == "k4_s0"
    assert case["choice"]["evaluation_arms"] == ["k4_s0"]
    assert "evaluation" not in failed
    model_path = run.failed / "model.pkl"
    digest = run.audit.gbm.file_hash(model_path)
    assert failed["unevaluated_model_artifact"] == {
        "sha256": digest,
        "bytes": model_path.stat().st_size,
        "trusted_for_evaluation": False,
    }
    assert summary["raw_artifact_sha256"][f"{WORKER_KEY}/model.pkl"] == digest
