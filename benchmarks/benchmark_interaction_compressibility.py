"""Admission helpers for the fixed saved-model compression pilot.

No model fitting or test-partition evaluation belongs in this diagnostic.
The eventual worker must call load_reference_case before using saved models.
Runtime imports are delayed so --source-root can select the frozen checkout
even when this new diagnostic lives in another checkout.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("interaction_compressibility_cases.json")
MEASUREMENT_SHA256 = "45e1934e516b4b2db3aadc0b9072d5b81bc656912c95d6a25aa631fbb942f7fc"
CASES = {"uci_airfoil": "k4_s2", "uci_concrete": "k6_s4", "kaggle_king_county_sales": "k6_s4"}


def read_manifest() -> dict:
    manifest = json.loads(MANIFEST.read_text())
    if (
        manifest["schema_version"] != 1
        or manifest["cases"] != CASES
        or manifest["partitions"] != ["train", "valid"]
        or manifest["fit_allowed"] is not False
        or manifest["test_evaluation_allowed"] is not False
        or manifest["measurement_sha256"] != MEASUREMENT_SHA256
    ):
        raise ValueError("Manifest differs from the fixed development-only pilot")
    return manifest


def read_measurement(manifest: dict, artifact_root: Path) -> dict:
    payload = (Path(artifact_root) / manifest["measurement_path"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != MEASUREMENT_SHA256:
        raise ValueError("Frozen measurement hash differs")
    return json.loads(payload)


def _runtime_identity(runtime: dict) -> dict:
    # Library locations may move with the checkout; versions and backend
    # configuration, including CPU architecture and thread counts, may not.
    return {
        **runtime,
        "threadpools": [
            {key: value for key, value in pool.items() if key != "filepath"}
            for pool in runtime["threadpools"]
        ],
    }


def admit_runtime(source_root: Path, measurement: dict) -> dict:
    """Check frozen code, actual imports and the one-thread environment before pickle."""
    root = Path(source_root).resolve()
    sys.path[:0] = [str(root / "src"), str(root / "benchmarks")]
    package = importlib.import_module("superglm")
    package_root = root / "src" / "superglm"
    if Path(package.__file__).resolve().parent != package_root:
        raise ValueError("The imported package is outside the requested source root")
    # A prior import must not mix submodules from a different checkout.
    for name, module in tuple(sys.modules.items()):
        if name.startswith("superglm.") and getattr(module, "__file__", None):
            if not Path(module.__file__).resolve().is_relative_to(package_root):
                raise ValueError(f"The imported {name} is outside the requested source root")
    expected = measurement["protocol"]["source"]
    names = [*expected["existing"]["benchmark_files_sha256"], *expected["new_files"]]
    for filename in names:
        module = importlib.import_module(Path(filename).stem)
        if Path(module.__file__).resolve() != root / "benchmarks" / filename:
            raise ValueError(f"The imported {filename} is outside the requested source root")
    broad = importlib.import_module("benchmark_broad_interactions")
    housing = importlib.import_module("benchmark_housing_tensor")
    actual = broad.source_identity()
    if (
        actual != expected
        or housing.source_fingerprint() != expected["existing"]["package_source_sha256"]
    ):
        raise ValueError("Frozen source identity differs")
    runtime = broad.gbm.runtime()
    if _runtime_identity(runtime) != _runtime_identity(measurement["runtime"]):
        raise ValueError("Frozen numerical runtime differs")
    return {"source_root": str(root), "source": actual, "runtime": runtime}


def admit_model(model, pairs: list) -> None:
    """Refuse terms or response maps outside the validated Gaussian tensor case."""
    from interaction_compression_geometry import fitted_term_beta

    from superglm import Gaussian, IdentityLink

    if type(model._distribution) is not Gaussian or type(model._link) is not IdentityLink:
        raise ValueError("Only Gaussian models with the exact identity link are admitted")
    names = [f"{left}:{right}" for left, right in pairs]
    if list(model._interaction_specs) != names:
        raise ValueError("Saved model pair order differs from the frozen selection")
    for name, pair in zip(names, pairs, strict=True):
        spec = model._interaction_specs[name]
        fitted_term_beta(model, name)
        if list(spec.parent_names) != pair:
            raise ValueError("Saved term parents differ from the frozen pair order")


def development_partition(prepared: dict, name: str) -> dict:
    """Expose raw units, frozen model inputs, responses and weights for train/valid."""
    if name not in ("train", "valid"):
        raise ValueError("Only train and valid partitions may be accessed")
    import benchmark_broad_interactions as broad
    import benchmark_real_interactions as base

    raw, response, weights = broad.partition(prepared, name)
    return {
        "raw": raw,
        "features": base.transform_features(raw, prepared["state"]),
        "response": response,
        "weights": weights,
    }


def load_reference_case(
    dataset: str, *, source_root: Path, artifact_root: Path = REPO, data_root: Path | None = None
) -> dict:
    """Load only the fixed selected model and its needed additive controls.

    The frozen adapter reads the raw source to reproduce its data identity.
    Only training and validation partitions leave this function. All source,
    data and artifact-byte checks finish before the first pickle is decoded.
    """
    manifest = read_manifest()
    if dataset not in manifest["cases"]:
        raise ValueError("Dataset is outside the fixed selected cases")
    measurement = read_measurement(manifest, artifact_root)
    runtime = admit_runtime(source_root, measurement)
    import broad_interaction_data as data

    case = measurement["datasets"][dataset]
    choice = {
        key: case["choice"][key] for key in ("chosen_arm", "additive_arm", "matching_additive_arm")
    }
    if choice["chosen_arm"] != manifest["cases"][dataset]:
        raise ValueError("Selected arm differs from the fixed case")
    arms = list(dict.fromkeys(choice.values()))
    if any(
        arm is None or (arm != choice["chosen_arm"] and not arm.endswith("_s0")) for arm in arms
    ):
        raise ValueError("Required additive controls are unavailable")
    prepared = data.load_prepared(
        dataset, data_root=data.DEFAULT_ROOT if data_root is None else data_root
    )

    def identity(metadata):
        return {
            key: value
            for key, value in metadata.items()
            if key not in ("source_path", "source_registry")
        }

    if (
        identity(prepared["metadata"]) != identity(case["data"])
        or prepared["metadata"].get("source_bytes_verified") is not True
    ):
        raise ValueError("Frozen data identity differs or source bytes are unverified")
    if prepared["metadata"]["family"] != "gaussian":
        raise ValueError("Only Gaussian data are admitted")
    payloads, records = {}, {}
    for arm in arms:
        record = case["fits"][arm]
        if record["status"] != "converged" or (arm != choice["chosen_arm"] and record["pairs"]):
            raise ValueError(
                "Saved reference is not a converged selected model or additive control"
            )
        path = Path(artifact_root) / manifest["run_root"] / dataset / arm / "model.pkl"
        payload = path.read_bytes()
        if (
            hashlib.sha256(payload).hexdigest() != record["model_pickle_sha256"]
            or len(payload) != record["model_pickle_bytes"]
        ):
            raise ValueError(f"Saved model hash or byte count differs: {dataset}/{arm}")
        payloads[arm] = payload
        records[arm] = {
            key: record[key]
            for key in ("pairs", "model_pickle_sha256", "model_pickle_bytes", "validation")
        }
    models = {}
    for arm, payload in payloads.items():
        model = pickle.loads(payload)
        admit_model(model, records[arm]["pairs"])
        models[arm] = model
    return {
        "dataset": dataset,
        "choice": choice,
        "models": models,
        "records": records,
        "data": prepared["metadata"],
        "preprocessing": prepared["state"],
        "runtime": runtime,
        "artifact_root": str(Path(artifact_root).resolve()),
        "raw_feature_units": "original source units",
        "score_scale": "identity_link",
        "partitions": {
            name: development_partition(prepared, name) for name in manifest["partitions"]
        },
    }


def replay_validation(model, partition: dict, recorded: dict) -> dict:
    """Require exact replay of the recorded loss in the admitted frozen runtime."""
    import benchmark_broad_interactions as broad

    replayed = broad.score(
        partition["response"],
        model.predict(partition["features"]),
        "gaussian",
        partition["weights"],
    )
    if replayed != recorded:
        raise ValueError(f"Recorded validation loss replay differs: {replayed} != {recorded}")
    return {
        "recorded": recorded,
        "replayed": replayed,
        "absolute_loss_difference": abs(replayed["primary_loss"] - recorded["primary_loss"]),
    }
