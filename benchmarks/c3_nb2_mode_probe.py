"""Untimed fixed-penalty consistency probe from a saved NB2 benchmark mode."""

from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal, localcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from c3_c1_complete_fit import data_fixture
from scipy.special import gammaln

from superglm._frame import as_eager_frame
from superglm.distributional.families.negative_binomial import NegativeBinomialLS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.kernels import negative_binomial as nb_kernel
from superglm.distributional.kernels.negative_binomial import _MIN_RATIO
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.predictor import compile_predictors
from superglm.distributional.smoothing.endpoint_direction import (
    finite_difference_curvature_direction,
    finite_difference_curvature_second_direction,
)
from superglm.distributional.solver.chunks import (
    evaluate_chunked_log_likelihood,
    materialize_terminal_predictions,
)
from superglm.distributional.solver.solver import (
    _evaluate_state_unmeasured,
    _geometry,
    _validated_context,
)
from superglm.distributional.weights import WeightContract, resolve_likelihood_weights
from superglm.links import LogLink


def decimal_zero_count_curvature(eta):
    mean, size = (value.exp() for value in eta)
    ratio = mean / size
    denominator = 1 + ratio
    return (
        mean / denominator**2,
        size * ratio**2 / denominator**2,
        size * (denominator.ln() - ratio / denominator - ratio**2 / denominator**2),
    )


def row_derivative_probe(arrays):
    """Audit existing stencils on the refused row, bypassing only its domain guard."""
    family = NegativeBinomialLS()
    y = np.zeros(1)
    weights = resolve_likelihood_weights(None, n_observations=1, contract=WeightContract("prior"))
    plan = family.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    links = (LogLink(), LogLink())
    theta = arrays["train_parameters"][96072].copy()
    report = {
        "row": 96072,
        "count": 0,
        "guard_bypass": "diagnostic only; original restored",
        "probes": [],
    }
    for multiplier in (1.0, 10.0, 10000.0):
        eta = np.log(theta * np.array([1.0, multiplier]))[None, :]
        for axis in (0, 1):
            direction = np.zeros((1, 2))
            direction[:, axis] = 1.0
            rejection = None
            try:
                finite_difference_curvature_direction(family, y, eta, direction, links, plan)
            except Exception as exc:
                rejection = {"type": type(exc).__name__, "message": str(exc)}
            original_ratio = nb_kernel._MIN_RATIO
            try:
                nb_kernel._MIN_RATIO = 0.0
                first = finite_difference_curvature_direction(
                    family, y, eta, direction, links, plan
                )
                second = finite_difference_curvature_second_direction(
                    family, y, eta, direction, direction, links, plan
                )
            finally:
                nb_kernel._MIN_RATIO = original_ratio
            with localcontext() as context:
                context.prec = 100
                center = [Decimal.from_float(value) for value in eta[0]]
                step = Decimal("1e-15")
                plus, minus = center.copy(), center.copy()
                plus[axis] += step
                minus[axis] -= step
                c, p, m = map(decimal_zero_count_curvature, (center, plus, minus))
                expected_first = np.array([float((a - b) / (2 * step)) for a, b in zip(p, m)])
                expected_second = np.array(
                    [float((a - 2 * b + c_) / step**2) for a, b, c_ in zip(p, c, m)]
                )
            first_error = np.abs(first.values[0] - expected_first)
            second_error = np.abs(second.values[0] - expected_second)
            report["probes"].append(
                {
                    "theta_multiplier": multiplier,
                    "axis": axis,
                    "original_guard_rejection": rejection,
                    "third_values": first.values[0].tolist(),
                    "third_reference": expected_first.tolist(),
                    "third_error": first_error.tolist(),
                    "third_certificate": first.certificate[0].tolist(),
                    "fourth_values": second.values[0].tolist(),
                    "fourth_reference": expected_second.tolist(),
                    "fourth_error": second_error.tolist(),
                    "fourth_certificate": second.certificate[0].tolist(),
                }
            )
    return report


def stable_count_log_likelihood(y, theta):
    """Independent exact-count recurrence, accumulated in extended precision."""
    mean = theta[:, 0].astype(np.longdouble)
    size = theta[:, 1].astype(np.longdouble)
    count = y.astype(np.int64)
    ratio = mean / size
    rows = count * np.log(mean) - (count + size) * np.log1p(ratio)
    for offset in range(1, int(count.max())):
        selected = count > offset
        rows[selected] += np.log1p(np.longdouble(offset) / size[selected])
    rows -= gammaln(count + 1)
    return float(np.sum(rows, dtype=np.longdouble)), np.asarray(rows, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(".benchmark-artifacts/c3-practical/nb2-recovery/recovery-0.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(".benchmark-artifacts/c3-practical/nb2-mode-probe/probe.json"),
    )
    parser.add_argument("--row-derivatives-only", action="store_true")
    args = parser.parse_args()
    saved = json.loads(args.input.read_text())
    arrays = np.load(args.input.with_suffix(".npz"))
    if args.row_derivatives_only:
        report = row_derivative_probe(arrays)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        return
    fixture_args = SimpleNamespace(**saved["config"])
    fixture_args.data = Path(fixture_args.data)
    model, frame, y, weight, offsets, _holdout, _holdout_offsets, _provenance = data_fixture(
        fixture_args
    )
    fingerprints = {
        "train_frame": hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest(),
        "y": hashlib.sha256(y.tobytes()).hexdigest(),
        "offset:mean": hashlib.sha256(offsets["mean"].tobytes()).hexdigest(),
    }
    for key, value in fingerprints.items():
        assert value == saved["fixture"]["fingerprints"][key], key
    weights = resolve_likelihood_weights(
        weight, n_observations=len(y), contract=WeightContract("prior")
    )
    family = model.family
    plan = family.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    compiled = compile_predictors(
        as_eager_frame(frame),
        weights,
        family.parameters,
        model.predictors,
        offsets=offsets,
        model_discrete=fixture_args.discrete,
        n_bins_config=fixture_args.n_bins,
    )
    layout = build_stacked_layout(compiled)
    beta = arrays["coefficients"]
    lambdas = saved["result"]["smoothing_parameters"]
    penalty = layout.penalty_matrix(lambdas)
    chunk_size = saved["coefficient_fits"][-1]["resolved_chunk_size"]
    context = _validated_context(
        family,
        layout,
        y,
        plan,
        penalty,
        coefficient_curvature="observed",
        chunk_size=chunk_size,
        coefficient_face=None,
    )
    state = _evaluate_state_unmeasured(context, beta)
    assert state is not None
    geometry = _geometry(context, state, "observed")
    gradient = geometry.score_penalized
    direction = arrays["covariance"] @ arrays["terminal_score"]
    minimum_mean_theta_ratio = getattr(nb_kernel, "_MIN_MEAN_THETA_RATIO", _MIN_RATIO)
    report = {
        "timing_status": "unmeasured diagnostic; no smoothing refits",
        "input": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "fingerprints": fingerprints,
        "lambdas": lambdas,
        "saved_gradient_l2": float(np.linalg.norm(arrays["terminal_score"])),
        "fresh_gradient_l2": float(np.linalg.norm(gradient)),
        "fresh_saved_gradient_max_difference": float(
            np.max(np.abs(gradient - arrays["terminal_score"]))
        ),
        "predicted_directional_slope": float(gradient @ direction),
        "saved_directional_slope": float(arrays["terminal_score"] @ direction),
        "full_newton_relative_correction": float(
            np.max(np.abs(direction)) / (1 + np.max(np.abs(beta)))
        ),
        "minimum_supported_mean_theta_ratio": minimum_mean_theta_ratio,
        "curve": [],
    }
    for step in [0.0, 1.0, -1.0, 0.5, 0.1, -0.1, 0.01, -0.01, 0.001, -0.001, 2.0**-28]:
        probe_beta = beta + step * direction
        probe = _evaluate_state_unmeasured(context, probe_beta, derivative_order=0)
        rejection = None
        if probe is None:
            try:
                evaluate_chunked_log_likelihood(
                    family, layout, y, plan, probe_beta, chunk_size=chunk_size
                )
            except Exception as exc:
                rejection = {"type": type(exc).__name__, "message": str(exc)}
        _eta, theta = materialize_terminal_predictions(layout, probe_beta, chunk_size=chunk_size)
        ratios = theta[:, 0] / theta[:, 1]
        boundary_row = int(np.argmin(ratios))
        stable_total, stable_rows = stable_count_log_likelihood(y, theta)
        penalty_value = 0.5 * float(probe_beta @ penalty @ probe_beta)
        try:
            natural = family.evaluate_natural(y, theta, plan, derivative_order=0)
            package_rows = natural.optimizing_log_likelihood + natural.parameter_independent_carrier
            row_error = package_rows - stable_rows
        except Exception as exc:
            package_rows = None
            row_error = None
            rejection = {"type": type(exc).__name__, "message": str(exc)}
        entry = {
            "step": step,
            "rejection": rejection,
            "optimizing_penalized": None
            if probe is None
            else probe.penalized_optimizing_log_likelihood,
            "reported_penalized": None if probe is None else probe.penalized_log_likelihood,
            "stable_reported_penalized": stable_total - penalty_value,
            "package_vs_stable_likelihood": None
            if probe is None
            else probe.log_likelihood - stable_total,
            "max_row_likelihood_error": None
            if row_error is None
            else float(np.max(np.abs(row_error))),
            "sum_row_likelihood_error": None if row_error is None else float(np.sum(row_error)),
            "theta_max": float(theta[:, 1].max()),
            "boundary_row": boundary_row,
            "boundary_count": float(y[boundary_row]),
            "boundary_mean": float(theta[boundary_row, 0]),
            "boundary_theta": float(theta[boundary_row, 1]),
            "boundary_ratio": float(ratios[boundary_row]),
            "boundary_relative_margin": float(
                ratios[boundary_row] / minimum_mean_theta_ratio - 1.0
            ),
            "former_boundary_relative_margin": float(ratios[boundary_row] / _MIN_RATIO - 1.0),
        }
        if step == 0.0:
            assert row_error is not None and package_rows is not None
            report["prediction_max_difference"] = float(
                np.max(np.abs(theta - arrays["train_parameters"]))
            )
            order = np.argsort(np.abs(row_error))[-10:][::-1]
            report["largest_row_errors"] = [
                {
                    "row": int(index),
                    "y": float(y[index]),
                    "mean": float(theta[index, 0]),
                    "theta": float(theta[index, 1]),
                    "package": float(package_rows[index]),
                    "stable": float(stable_rows[index]),
                }
                for index in order
            ]
        report["curve"].append(entry)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(entry), flush=True)
    curve = {entry["step"]: entry for entry in report["curve"]}
    baseline = curve[0.0]["stable_reported_penalized"]
    report["stable_objective_direction_checks"] = {
        "quadratic_full_step_gain": 0.5 * float(gradient @ direction),
        "actual_full_step_gain": curve[1.0]["stable_reported_penalized"] - baseline,
        "central_slopes": {
            str(step): (
                curve[step]["stable_reported_penalized"] - curve[-step]["stable_reported_penalized"]
            )
            / (2.0 * step)
            for step in (0.1, 0.01, 0.001)
        },
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"curve", "largest_row_errors"}
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
