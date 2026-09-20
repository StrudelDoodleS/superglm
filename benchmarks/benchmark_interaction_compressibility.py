"""Bounded, saved-model interaction compression pilot.

No model fitting or test-partition evaluation belongs in this diagnostic.
Every worker calls load_reference_case before using saved models.
Runtime imports are delayed so --source-root can select the frozen checkout
even when this new diagnostic lives in another checkout.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import pickle
import resource
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("interaction_compressibility_cases.json")
MEASUREMENT_SHA256 = "45e1934e516b4b2db3aadc0b9072d5b81bc656912c95d6a25aa631fbb942f7fc"
CASES = {"uci_airfoil": "k4_s2", "uci_concrete": "k6_s4", "kaggle_king_county_sales": "k6_s4"}


def read_manifest(path: Path = MANIFEST) -> dict:
    manifest = json.loads(Path(path).read_text())
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
        "row_ids": prepared["rows"][name].copy(),
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
            for key in (
                "pairs",
                "model_pickle_sha256",
                "model_pickle_bytes",
                "validation",
                "fit_seconds",
                "fit_end_peak_process_rss_mib",
                "retained_model_storage",
                "resolved_direct_backend",
                "backend_group_counts",
                "worker_seconds",
                "worker_end_peak_process_rss_mib",
                "process_seconds",
            )
            if key in record
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


SCOPE_FLAGS = {
    "numerical_certificate": False,
    "statistical_uncertainty_estimated": False,
    "new_model_fits": 0,
    "fresh_test_evaluation": False,
    "fitting_speedup_claim": False,
}


def loss_arithmetic(response, prediction, weights) -> dict:
    """Weighted Gaussian MSE with the supplied estimated arithmetic allowance."""
    from interaction_compression_geometry import _finite_array, _gamma

    y, p, w = (_finite_array(value, "loss input") for value in (response, prediction, weights))
    if y.ndim != 1 or y.shape != p.shape or y.shape != w.shape or not len(y):
        raise ValueError("Loss inputs must be nonempty equal-length vectors")
    total = float(w.sum())
    if np.any(w < 0) or not np.isfinite(total) or total <= 0:
        raise ValueError("Loss weights must have finite positive sum")
    alpha, beta = _gamma(len(y) + 5), _gamma(len(y))
    if beta >= 1:
        raise ValueError("Loss arithmetic dimension is unsupported")
    rho = (alpha + beta) / (1 - beta)
    if rho >= 1:
        raise ValueError("Loss arithmetic dimension is unsupported")
    loss = float(np.average((y - p) ** 2, weights=w))
    allowance = rho / (1 - rho) * loss
    if not np.isfinite([loss, allowance]).all():
        raise ValueError("Loss arithmetic scale is unsupported")
    return {"loss": loss, "loss_arithmetic_allowance": allowance}


def _weighted_norm_upper(vector, weights) -> float:
    result = loss_arithmetic(np.zeros_like(vector), vector, weights)
    return float(
        np.sqrt(result["loss"] + result["loss_arithmetic_allowance"])
        / (1 - np.finfo(float).eps / 2)
    )


def gain_retention(
    *,
    additive_loss,
    reference_loss,
    candidate_loss,
    additive_allowance,
    reference_allowance,
    candidate_allowance,
) -> dict:
    """Keep raw ratios only when positive gain clears its arithmetic band."""
    from interaction_compression_geometry import _gamma

    u = np.finfo(float).eps / 2
    D, N = additive_loss - reference_loss, additive_loss - candidate_loss
    B_D = additive_allowance + reference_allowance + _gamma(1) * abs(D)
    B_N = additive_allowance + candidate_allowance + _gamma(1) * abs(N)
    result = {
        "gain_denominator": D,
        "gain_denominator_allowance": B_D,
        "candidate_gain": N,
        "candidate_gain_allowance": B_N,
        "candidate_minus_reference_loss": candidate_loss - reference_loss,
        "gain_retention": None,
        "gain_retention_allowance": None,
        "gain_ratio_status": "denominator_not_positive_beyond_allowance",
    }
    if D > B_D:
        R = N / D
        B_R = (B_N + abs(R) / (1 - u) * B_D) / (D - B_D) + _gamma(1) * abs(R)
        if np.isfinite([R, B_R]).all():
            result.update(
                gain_retention=R, gain_retention_allowance=B_R, gain_ratio_status="admitted"
            )
        else:
            result["gain_ratio_status"] = "nonfinite_ratio_or_allowance"
    return result


def replace_interactions(reference_prediction, replacements: list[dict]) -> dict:
    """Apply the declared term order to a copy of the shared stored predictor."""
    from interaction_compression_geometry import (
        _finite_array,
        _gamma,
        contraction_roundoff_estimate,
        score_pairs,
    )

    prediction = reference_prediction.copy()
    v_num, v_full = np.zeros_like(prediction), np.zeros_like(prediction)
    for term in replacements:
        left, right = term["left"], term["right"]
        C, candidate = term["reference_coefficients"], term["candidate_coefficients"]
        effect = score_pairs(left, candidate, right)
        delta = effect - term["reference_effect"]
        prediction += delta
        v_num += (
            contraction_roundoff_estimate(left, C, right)
            + contraction_roundoff_estimate(left, candidate, right)
            + term["runtime_discrepancy"]
            + _gamma(1) * (np.abs(delta) + np.abs(prediction))
        )
        v_full += (
            np.linalg.norm(left, axis=1)
            * np.linalg.norm(right, axis=1)
            * term["coefficient_allowance"]
        )
    return {
        "prediction": _finite_array(prediction, "updated predictor"),
        "predictor_update_allowance": _finite_array(v_num, "predictor arithmetic allowance"),
        "coefficient_effect_allowance": _finite_array(v_full, "coefficient effect allowance"),
        "full_budget_predictor_allowance": _finite_array(
            (1 + _gamma(1)) * (v_num + v_full), "full predictor allowance"
        ),
    }


def _partition_metrics(partition, baseline, additive, update, full_budget) -> dict:
    from interaction_compression_geometry import _gamma

    y, w = partition["response"], partition["weights"]
    p, p0 = update["prediction"], baseline
    current, reference = loss_arithmetic(y, p, w), loss_arithmetic(y, p0, w)
    e = _weighted_norm_upper(update["predictor_update_allowance"], w)
    rnorm = _weighted_norm_upper(y - p, w)
    candidate_allowance = current["loss_arithmetic_allowance"] + 2 * rnorm * e + e * e
    if not np.isfinite(candidate_allowance):
        raise ValueError("Candidate loss allowance must be finite")
    delta = p - p0
    result = {
        **current,
        "reference_loss": reference["loss"],
        "reference_loss_arithmetic_allowance": reference["loss_arithmetic_allowance"],
        "candidate_loss_allowance": candidate_allowance,
        "paired_weighted_prediction_error": loss_arithmetic(p0, p, w)["loss"],
        "maximum_paired_discrepancy": float(np.max(np.abs(delta), initial=0)),
        "full_budget_loss_allowance": None,
        "full_budget_replay_passed": None,
        "additive_controls": {},
    }
    for name in (
        "predictor_update_allowance",
        "coefficient_effect_allowance",
        "full_budget_predictor_allowance",
    ):
        result[name] = {
            "maximum": float(np.max(update[name], initial=0)),
            "weighted_norm_upper": _weighted_norm_upper(update[name], w),
        }
    if full_budget:
        e_full = _weighted_norm_upper(
            update["predictor_update_allowance"] + update["coefficient_effect_allowance"], w
        )
        B = 2 * _weighted_norm_upper(y - p0, w) * e_full + e_full**2
        B += reference["loss_arithmetic_allowance"] + current["loss_arithmetic_allowance"]
        B += _gamma(1) * abs(current["loss"] - reference["loss"])
        if not np.isfinite(B):
            raise ValueError("Full-budget loss allowance must be finite")
        passed = np.all(np.abs(delta) <= update["full_budget_predictor_allowance"])
        passed = bool(passed and abs(current["loss"] - reference["loss"]) <= B)
        result.update(full_budget_loss_allowance=float(B), full_budget_replay_passed=passed)
        if not passed:
            raise ValueError(
                "Full-budget predictor/loss reconstruction exceeds its estimated allowance"
            )
    for role, control in additive.items():
        score = loss_arithmetic(y, control["prediction"], w)
        result["additive_controls"][role] = {
            "arm": control["arm"],
            **score,
            **gain_retention(
                additive_loss=score["loss"],
                reference_loss=reference["loss"],
                candidate_loss=current["loss"],
                additive_allowance=score["loss_arithmetic_allowance"],
                reference_allowance=reference["loss_arithmetic_allowance"],
                candidate_allowance=candidate_allowance,
            ),
        }
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, result: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, allow_nan=False, default=lambda value: value.item()) + "\n"
    )
    temporary.replace(path)


def _prepare_terms(case, arrays) -> tuple[dict, dict]:
    """Freeze each saved term's training QR metric and penalty modes once."""
    import interaction_compression_geometry as geometry

    model = case["models"][case["choice"]["chosen_arm"]]
    terms, metadata = {}, {}
    for index, (name, spec) in enumerate(model._interaction_specs.items()):
        key = f"term_{index}"
        beta = geometry.fitted_term_beta(model, name)
        C = geometry.effective_tensor_coefficients(spec, beta)
        term = {"name": name, "coefficients": C, "partitions": {}, "refusal_reason": None}
        meta = {"name": name, "shape": list(C.shape), "original_coefficient_entries": C.size}
        arrays[f"{key}__coefficients"] = C
        try:
            for part, partition in case["partitions"].items():
                left_name, right_name = spec.parent_names
                x1 = partition["features"][left_name].to_numpy()
                x2 = partition["features"][right_name].to_numpy()
                left, right = geometry.paired_centered_bases(spec, x1, x2)
                effect = spec.score(x1, x2, beta)
                discrepancy = np.abs(effect - geometry.score_pairs(left, C, right))
                roundoff = geometry.contraction_roundoff_estimate(left, C, right)
                if not np.all(discrepancy <= roundoff):
                    raise ValueError("Saved term runtime replay exceeds the contraction allowance")
                term["partitions"][part] = {
                    "left": left,
                    "right": right,
                    "reference_effect": effect,
                    "runtime_discrepancy": discrepancy,
                }
                for label, value in term["partitions"][part].items():
                    arrays[f"{key}__{part}__{label}"] = value
            train = term["partitions"]["train"]
            try:
                metrics = geometry.product_metric_factors(
                    train["left"], train["right"], case["partitions"]["train"]["weights"]
                )
                term["metrics"] = metrics
                eta_left, eta_right = [
                    metric.diagnostics["metric_relative_error"] for metric in metrics
                ]
                meta["product_metric_relative_discrepancy"] = (
                    eta_left + eta_right + eta_left * eta_right
                )
                meta["metric_diagnostics"] = [metric.diagnostics for metric in metrics]
                for label, metric in zip(("left", "right"), metrics, strict=True):
                    arrays[f"{key}__{label}_metric_factor"] = metric.factor
            except ValueError as exc:
                term["metric_refusal"] = meta["metric_refusal"] = str(exc)
            try:
                modes = geometry.saved_penalty_modes(spec)
                term["modes"] = modes
                meta["mode_diagnostics"] = [mode.diagnostics for mode in modes]
                for label, mode in zip(("left", "right"), modes, strict=True):
                    arrays[f"{key}__{label}_modes"] = mode.modes
                    arrays[f"{key}__{label}_eigenvalues"] = mode.eigenvalues
            except ValueError as exc:
                term["modal_refusal"] = meta["modal_refusal"] = str(exc)
        except ValueError as exc:
            term["refusal_reason"] = meta["refusal_reason"] = str(exc)
        terms[key], metadata[key] = term, meta
    return terms, metadata


def _term_candidate(term, method, budget) -> tuple[np.ndarray, dict]:
    import interaction_compression_geometry as geometry

    C = term["coefficients"]
    p, q = C.shape
    if term["refusal_reason"]:
        raise ValueError(term["refusal_reason"])
    full = budget >= (min(p, q) if method == "rank" else max(p, q))
    details = {
        "coefficient_reconstruction_allowance": 0.0,
        "spectral_tail": None,
        "product_measure_squared_error": None,
        "product_metric_refusal": term.get("metric_refusal"),
        "full_budget": full,
    }
    if method == "rank":
        if "metrics" not in term:
            raise ValueError(term["metric_refusal"])
        L, R = term["metrics"]
        result = geometry.truncate_in_product_metric(C, L.factor, R.factor, min(budget, p, q))
        candidate = result.coefficients
        details.update(
            spectral_tail=result.discarded_product_energy,
            product_measure_squared_error=result.realized_product_energy,
            spectral_realized_energy_allowance=result.energy_allowance,
            # Factor conditioning is stored once with the reference term. The
            # candidate retains its smaller, width-based joint tolerance.
            diagnostics={
                key: value
                for key, value in result.diagnostics.items()
                if key not in ("left_factor", "right_factor")
            },
        )
        if full:
            d = result.diagnostics
            b_H = (
                d["whitening_allowance"]
                + d["svd_residual"]
                + d["kept_product_allowance"]
                + d["recovery_residual"]
            )
            details["coefficient_reconstruction_allowance"] = b_H / (
                L.diagnostics["minimum_singular_value"] * R.diagnostics["minimum_singular_value"]
            )
    else:
        if "modes" not in term:
            raise ValueError(term["modal_refusal"])
        U, V = [mode.modes for mode in term["modes"]]
        if full:
            train = term["partitions"]["train"]
            candidate, diagnostic = geometry.modal_full_reconstruction(
                C, U, V, train["left"], train["right"]
            )
            details["coefficient_reconstruction_allowance"] = diagnostic["coefficient_allowance"]
            details["diagnostics"] = diagnostic
        else:
            candidate = geometry.modal_prefix(C, U, V, budget)
        if "metrics" in term:
            L, R = term["metrics"]
            details["product_measure_squared_error"] = float(
                np.linalg.norm(L.factor @ (C - candidate) @ R.factor.T) ** 2
            )
    if not np.isfinite(details["coefficient_reconstruction_allowance"]):
        raise ValueError("Nonfinite full-budget coefficient allowance")
    return candidate, details


def diagnostic_worker(dataset: str, *, source_root: Path, output: Path) -> dict:
    """Evaluate every admitted budget, retaining refusals and no-fit evidence."""
    start = time.perf_counter()
    case = load_reference_case(dataset, source_root=source_root)
    from benchmark_housing_tensor import retained_model_storage

    arrays, baseline, additive, replay = {}, {}, {}, {}
    for arm, model in case["models"].items():
        replay[arm] = replay_validation(
            model, case["partitions"]["valid"], case["records"][arm]["validation"]
        )
    for part, partition in case["partitions"].items():
        for name in ("response", "weights", "row_ids"):
            arrays[f"{part}__{name}"] = partition[name]
        for kind in ("raw", "features"):
            for index, name in enumerate(partition[kind].columns):
                value = partition[kind][name].to_numpy()
                arrays[f"{part}__{kind}_{index}"] = (
                    value.astype(str) if value.dtype.hasobject else value
                )
        for arm, model in case["models"].items():
            arrays[f"{part}__{arm}__prediction"] = model.predict(partition["features"])
        baseline[part] = arrays[f"{part}__{case['choice']['chosen_arm']}__prediction"]
        additive[part] = {
            role: {
                "arm": case["choice"][role],
                "prediction": arrays[f"{part}__{case['choice'][role]}__prediction"],
            }
            for role in ("matching_additive_arm", "additive_arm")
        }
    terms, term_metadata = _prepare_terms(case, arrays)
    reference_path = output / "reference.npz"
    np.savez_compressed(reference_path, **arrays)
    reference = {
        "dataset": dataset,
        "choice": case["choice"],
        "model_records": case["records"],
        "runtime": case["runtime"],
        "data": case["data"],
        "preprocessing": case["preprocessing"],
        "term_order": list(terms),
        "terms": term_metadata,
        "validation_replay": replay,
        "partitions": {
            part: {
                "rows": len(p["row_ids"]),
                "row_ids_sha256": hashlib.sha256(p["row_ids"].astype("<i8").tobytes()).hexdigest(),
                "raw_columns": list(p["raw"].columns),
                "feature_columns": list(p["features"].columns),
            }
            for part, p in case["partitions"].items()
        },
        "arrays_path": str(reference_path.relative_to(output.parent)),
        "arrays_sha256": _sha256(reference_path),
        "diagnostic_array_payload_bytes": sum(array.nbytes for array in arrays.values()),
        "selected_retained_model_storage": retained_model_storage(
            case["models"][case["choice"]["chosen_arm"]]
        ),
        "historical_test_designation": "already-used development data; no fresh access or evaluation",
        **SCOPE_FLAGS,
    }
    _write_json(output / "reference.json", reference)
    reference_hash = _sha256(output / "reference.json")
    result = {
        "dataset": dataset,
        "status": "running",
        "reference_path": str((output / "reference.json").relative_to(output.parent)),
        "reference_sha256": reference_hash,
        "attempts": [],
        **SCOPE_FLAGS,
    }
    scopes = [(key, [key]) for key in terms] + [("all_terms", list(terms))]
    result["expected_attempt_count"] = sum(
        1 + max(reduce(terms[key]["coefficients"].shape) for key in keys)
        for reduce in (min, max)
        for _, keys in scopes
    )
    _write_json(output / "case.json", result)
    for method in ("rank", "modal"):
        for scope, keys in scopes:
            limit = max(
                (min if method == "rank" else max)(terms[key]["coefficients"].shape) for key in keys
            )
            for budget in range(limit + 1):
                attempt_start = time.perf_counter()
                attempt_id = f"{method}__{scope}__{budget}"
                receipt = {
                    "attempt_id": attempt_id,
                    "dataset": dataset,
                    "method": method,
                    "scope": scope,
                    "requested_budget": budget,
                    "update_order": keys,
                    "reference_path": result["reference_path"],
                    "reference_sha256": reference_hash,
                    "status": "evaluated",
                    "refusal_reason": None,
                    "rank_zero_is_additive_refit": False,
                    "full_budget": budget == limit,
                    "terms": {},
                    "partitions": {},
                    **SCOPE_FLAGS,
                }
                diagnostic_arrays, candidates = {}, {}
                entries = factors = products = mode_columns = expanded = frozen_modes = 0
                for key in keys:
                    p, q = terms[key]["coefficients"].shape
                    rank, left_width, right_width = (
                        min(budget, p, q),
                        min(budget, p),
                        min(budget, q),
                    )
                    receipt["terms"][key] = {
                        "shape": [p, q],
                        "actual_rank_budget": rank if method == "rank" else None,
                        "selected_product_count": left_width * right_width
                        if method == "modal"
                        else None,
                        "selected_marginal_widths": [left_width, right_width]
                        if method == "modal"
                        else None,
                    }
                    entries += p * q
                    factors += rank * (p + q) if method == "rank" else 0
                    products += left_width * right_width if method == "modal" else 0
                    mode_columns += p * left_width + q * right_width if method == "modal" else 0
                    frozen_modes += (
                        p * p + q * q if method == "modal" and "modes" in terms[key] else 0
                    )
                try:
                    for key in keys:
                        candidate, details = _term_candidate(terms[key], method, budget)
                        candidates[key] = candidate
                        diagnostic_arrays[f"{key}__candidate_coefficients"] = candidate
                        expanded += candidate.nbytes
                        receipt["terms"][key].update(details)
                    for part, partition in case["partitions"].items():
                        replacements = [
                            {
                                **terms[key]["partitions"][part],
                                "reference_coefficients": terms[key]["coefficients"],
                                "candidate_coefficients": candidates[key],
                                "coefficient_allowance": receipt["terms"][key][
                                    "coefficient_reconstruction_allowance"
                                ],
                            }
                            for key in keys
                        ]
                        update = replace_interactions(baseline[part], replacements)
                        diagnostic_arrays.update(
                            {f"{part}__{name}": value for name, value in update.items()}
                        )
                        receipt["partitions"][part] = _partition_metrics(
                            partition,
                            baseline[part],
                            additive[part],
                            update,
                            receipt["full_budget"],
                        )
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                    receipt.update(status="refused", refusal_reason=str(exc))
                    for part in ("train", "valid"):
                        receipt["partitions"].setdefault(
                            part, {"status": "refused", "refusal_reason": str(exc)}
                        )
                receipt["storage"] = {
                    "original_coefficient_entries": entries,
                    "logical_factor_entries": factors,
                    "logical_selected_product_entries": products,
                    "logical_mode_column_entries": mode_columns,
                    "logical_candidate_payload_bytes": 8 * (factors + products + mode_columns),
                    "logical_payload_kind": "estimate, not allocated/exported factors or deployed model memory",
                    "expanded_candidate_coefficient_bytes": expanded,
                    "retained_frozen_modal_basis_payload_bytes": 8 * frozen_modes,
                    "diagnostic_array_payload_bytes": sum(
                        array.nbytes for array in diagnostic_arrays.values()
                    ),
                }
                attempt_arrays_path = output / f"{attempt_id}.npz"
                np.savez_compressed(attempt_arrays_path, **diagnostic_arrays)
                receipt.update(
                    arrays_path=str(attempt_arrays_path.relative_to(output.parent)),
                    arrays_sha256=_sha256(attempt_arrays_path),
                    diagnostic_seconds=time.perf_counter() - attempt_start,
                    process_high_water_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    / 1024,
                )
                receipt_path = output / f"{attempt_id}.json"
                _write_json(receipt_path, receipt)
                result["attempts"].append(
                    {
                        **receipt,
                        "receipt_path": str(receipt_path.relative_to(output.parent)),
                        "receipt_sha256": _sha256(receipt_path),
                    }
                )
                _write_json(output / "case.json", result)
    result["status"] = "evaluated"
    if all(attempt["status"] == "refused" for attempt in result["attempts"]):
        result["status"] = "no_admissible_candidate"
    result.update(
        worker_seconds=time.perf_counter() - start,
        process_high_water_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )
    _write_json(output / "case.json", result)
    return result


def aggregate_results(output: Path, processes: dict) -> dict:
    """Include case failures and every exported attempt in the batch receipt."""
    cases = {}
    for dataset, process in processes.items():
        path = output / dataset / "case.json"
        if path.exists():
            case = json.loads(path.read_text())
            case.update(
                case_receipt_path=str(path.relative_to(output)), case_receipt_sha256=_sha256(path)
            )
        else:
            case = {"dataset": dataset, "status": "case_failed_before_receipt", "attempts": []}
        cases[dataset] = {
            **case,
            "process": process,
            "unrun_attempt_count": case["expected_attempt_count"] - len(case["attempts"])
            if "expected_attempt_count" in case
            else None,
        }
    result = {
        "cases": cases,
        "attempt_count": sum(len(case["attempts"]) for case in cases.values()),
        "evaluated_count": sum(
            a["status"] == "evaluated" for case in cases.values() for a in case["attempts"]
        ),
        "refused_count": sum(
            a["status"] == "refused" for case in cases.values() for a in case["attempts"]
        ),
        "attempts_with_product_metric_refusal": sum(
            any(edge.get("product_metric_refusal") for edge in a["terms"].values())
            for case in cases.values()
            for a in case["attempts"]
        ),
        "status": "complete"
        if all(
            case["process"]["status"] == "success"
            and case["status"] in ("evaluated", "no_admissible_candidate")
            for case in cases.values()
        )
        else "incomplete",
        **SCOPE_FLAGS,
    }
    _write_json(output / "measurements.json", result)
    return result


def plot_results(result: dict, output: Path) -> list[dict]:
    """Plot every one-term and common-budget attempt, with visible refusals."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []
    for dataset, case in result["cases"].items():
        if not case["attempts"]:
            continue
        reference = json.loads((output / case["reference_path"]).read_text())
        scopes = list(reference["terms"]) + ["all_terms"]
        fig, axes = plt.subplots(
            len(scopes), 4, figsize=(18, 3.1 * len(scopes)), squeeze=False, constrained_layout=True
        )
        for row, scope in enumerate(scopes):
            selected = [a for a in case["attempts"] if a["scope"] == scope]
            title = (
                reference["terms"][scope]["name"] if scope != "all_terms" else "All selected terms"
            )
            axes[row, 0].set_ylabel(title.replace(":", " ×\n") + "\nSquared product error")
            for method, color, marker in (("rank", "#2166ac", "o"), ("modal", "#b35806", "s")):
                attempts = [
                    a for a in selected if a["method"] == method and a["status"] == "evaluated"
                ]
                x = [a["storage"]["logical_candidate_payload_bytes"] for a in attempts]
                product = [
                    sum(edge["product_measure_squared_error"] for edge in a["terms"].values())
                    if all(
                        edge["product_measure_squared_error"] is not None
                        for edge in a["terms"].values()
                    )
                    else np.nan
                    for a in attempts
                ]
                axes[row, 0].plot(x, product, color=color, marker=marker, label=method)
                for partition, style in (("train", "--"), ("valid", "-")):
                    axes[row, 1].plot(
                        x,
                        [
                            a["partitions"][partition]["paired_weighted_prediction_error"]
                            for a in attempts
                        ],
                        color=color,
                        marker=marker,
                        linestyle=style,
                        label=f"{method} {partition}",
                    )
                    axes[row, 2].plot(
                        x,
                        [a["partitions"][partition]["loss"] for a in attempts],
                        color=color,
                        marker=marker,
                        linestyle=style,
                        label=f"{method} {partition}",
                    )
                for role, style in (("matching_additive_arm", "-"), ("additive_arm", ":")):
                    gains = [
                        a["partitions"]["valid"]["additive_controls"][role]["gain_retention"]
                        for a in attempts
                    ]
                    axes[row, 3].plot(
                        x,
                        [np.nan if gain is None else gain for gain in gains],
                        color=color,
                        marker=marker,
                        linestyle=style,
                        label=f"{method} {'matching k' if role == 'matching_additive_arm' else 'best additive'}",
                    )
                refused = [
                    a for a in selected if a["method"] == method and a["status"] == "refused"
                ]
                if refused:
                    axes[row, 0].text(
                        0.02,
                        0.96 if method == "rank" else 0.84,
                        f"{method} refused budgets: "
                        + ", ".join(str(a["requested_budget"]) for a in refused),
                        transform=axes[row, 0].transAxes,
                        va="top",
                        color=color,
                        fontsize=8,
                    )
            evaluated = [a for a in selected if a["status"] == "evaluated"]
            if evaluated:
                metrics = evaluated[0]["partitions"]["valid"]
                axes[row, 2].axhline(
                    metrics["reference_loss"], color="black", linewidth=1, label="selected valid"
                )
                for role, style in (("matching_additive_arm", "--"), ("additive_arm", ":")):
                    axes[row, 2].axhline(
                        metrics["additive_controls"][role]["loss"],
                        color="gray",
                        linestyle=style,
                        linewidth=1,
                        label=f"{metrics['additive_controls'][role]['arm']} valid",
                    )
            for guide in (0.90, 0.95, 0.99):
                axes[row, 3].axhline(guide, color="gray", linewidth=0.6, linestyle="--")
            for column, label in enumerate(
                (
                    "Product error sum across replaced edges",
                    "Paired prediction MSE",
                    "Full-predictor MSE",
                    "Validation gain retention",
                )
            ):
                axes[row, column].set_title(label, fontsize=10)
                axes[row, column].set_xlabel("Logical candidate payload, bytes")
                axes[row, column].grid(alpha=0.2)
                axes[row, column].ticklabel_format(axis="both", style="sci", scilimits=(-3, 4))
                if row == 0:
                    axes[row, column].legend(fontsize=7)
            if not selected:
                axes[row, 0].text(
                    0.02,
                    0.96,
                    "Unrun after process failure",
                    transform=axes[row, 0].transAxes,
                    va="top",
                )
            replaced = reference["terms"] if scope == "all_terms" else [scope]
            dense_bytes = 8 * sum(
                reference["terms"][key]["original_coefficient_entries"] for key in replaced
            )
            for ax in axes[row]:
                ax.axvline(dense_bytes, color="gray", linewidth=0.7, linestyle=":")
        fig.suptitle(
            f"{dataset}: saved predictor compression\nDotted vertical line: original dense interaction bytes. Logical payload is not whole-model memory.\nGain guides 90/95/99% are provisional; no statistical uncertainty estimated.",
            fontsize=12,
        )
        path = output / f"interaction-compressibility-{dataset}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append({"path": path.name, "sha256": _sha256(path)})
    return figures


@contextmanager
def batch_deadline():
    """Interrupt Linux batch work at 900 seconds; restore the process alarm."""
    import signal
    import threading

    if sys.platform != "linux" or threading.current_thread() is not threading.main_thread():
        raise ValueError("The hard batch deadline requires the Linux main thread")
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer != (0.0, 0.0):
        raise ValueError("An existing process alarm prevents the hard batch deadline")
    previous_handler = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise TimeoutError("The 900-second batch deadline expired")

    signal.signal(signal.SIGALRM, expired)
    try:
        signal.setitimer(signal.ITIMER_REAL, 900)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


def run_batch(*, manifest_path: Path, source_root: Path, output: Path) -> dict:
    """Own three serial fresh processes, bounded through plotting and export."""
    import os
    import subprocess
    from datetime import UTC, datetime

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = read_manifest(manifest_path)
    source_root = source_root.resolve()
    sys.path[:0] = [str(source_root / "src"), str(source_root / "benchmarks")]
    started = time.perf_counter()
    metadata = {
        "started_utc": datetime.now(UTC).isoformat(),
        "raw_output": str(output),
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "pilot_code_sha256": {
            path.name: _sha256(path)
            for path in (
                Path(__file__).resolve(),
                Path(__file__).with_name("interaction_compression_geometry.py"),
            )
        },
        "manifest_sha256": _sha256(Path(manifest_path)),
        "source_root": str(source_root),
        "limits": {
            "case_process_seconds": 180,
            "batch_seconds": 900,
            "serial_workers": True,
            "batch_deadline_scope": "workers, plotting and export; terminal timeout receipts follow cleanup",
        },
        "expected_attempt_counts": {
            "uci_airfoil": 24,
            "uci_concrete": 60,
            "kaggle_king_county_sales": 60,
        },
    }
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "NUMBA_CACHE_DIR": str(REPO / ".cache" / "numba"),
    }
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        env[name] = "1"
    processes = {dataset: {"status": "unrun"} for dataset in manifest["cases"]}
    try:
        with batch_deadline():
            from benchmark_housing_tensor import run_isolated

            for dataset in manifest["cases"]:
                case_output = output / dataset
                case_output.mkdir()
                command = [
                    str(REPO / ".venv" / "bin" / "python"),
                    "-B",
                    str(Path(__file__).resolve()),
                    "--manifest",
                    str(Path(manifest_path).resolve()),
                    "--source-root",
                    str(source_root),
                    "--output",
                    str(case_output),
                    "--worker",
                    dataset,
                ]
                processes[dataset] = {"status": "running", "command": command}
                _write_json(case_output / "process.json", processes[dataset])
                try:
                    process = run_isolated(
                        command, log_path=case_output / "worker.log", timeout=180, env=env
                    )
                except TimeoutError:
                    raise  # The batch alarm is not a subprocess launch failure.
                except OSError as exc:
                    process = {"status": "launch_error", "reason": str(exc)}
                processes[dataset] = {**process, "command": command}
                _write_json(case_output / "process.json", processes[dataset])
                print(f"{dataset}: {process['status']}", flush=True)
            result = {**aggregate_results(output, processes), **metadata}
            if any(
                len(result["cases"][case]["attempts"]) != count
                for case, count in metadata["expected_attempt_counts"].items()
            ):
                result["status"] = "incomplete"
            result["figures"] = plot_results(result, output)
            _write_json(output / "measurements.json", result)
    except TimeoutError as exc:
        for dataset, process in processes.items():
            if process["status"] in ("running", "unrun"):
                process["status"] = (
                    "batch_timeout" if process["status"] == "running" else "unrun_batch_timeout"
                )
                if (output / dataset).exists():
                    _write_json(output / dataset / "process.json", process)
        result = {
            **aggregate_results(output, processes),
            **metadata,
            "status": "incomplete",
            "batch_timeout": str(exc),
            "figures": [
                {"path": path.name, "sha256": _sha256(path)}
                for path in output.glob("interaction-compressibility-*.png")
            ],
        }
    result.update(
        finished_utc=datetime.now(UTC).isoformat(), batch_seconds=time.perf_counter() - started
    )
    _write_json(output / "measurements.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "attempts": result["attempt_count"],
                "evaluated": result["evaluated_count"],
                "refused": result["refused_count"],
                "measurements": str(output / "measurements.json"),
            }
        ),
        flush=True,
    )
    return result


def main() -> int:
    import argparse
    import traceback

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=CASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        try:
            read_manifest(args.manifest)
            diagnostic_worker(args.worker, source_root=args.source_root, output=args.output)
            return 0
        except Exception as exc:
            receipt_path = args.output / "case.json"
            result = (
                json.loads(receipt_path.read_text()) if receipt_path.exists() else {"attempts": []}
            )
            _write_json(
                receipt_path,
                {
                    **result,
                    "dataset": args.worker,
                    "status": "case_failed",
                    "refusal_reason": f"{type(exc).__name__}: {exc}",
                    **SCOPE_FLAGS,
                },
            )
            traceback.print_exc()
            return 1
    return (
        0
        if run_batch(manifest_path=args.manifest, source_root=args.source_root, output=args.output)[
            "status"
        ]
        == "complete"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
