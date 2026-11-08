"""Complete-fit checks against independent objectives, separate from timings."""

import math
import pickle
from itertools import product

import mpmath as mp
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import fisk, genpareto, lomax

from superglm import GeneralizedParetoLSS, LogLogisticLSS, LomaxLSS, SuperLSS, s
from superglm._frame import as_eager_frame
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.predictor import Predictor, compile_predictors
from superglm.distributional.solver import DenseSolverConfig, fit_dense_fixed_lambda
from superglm.features import Numeric
from tests._declared_scalar_oracles import log_density
from tests.test_declared_scalar_adapter import weights


def independent_rows(name, y, eta):
    """Textbook log densities and log-parameter scores; no generated kernels."""
    scale, shape = np.exp(eta).T
    r = y / scale
    if name == "lomax":
        value = np.log(shape) - np.log(scale) - (shape + 1) * np.log1p(r)
        score = np.column_stack((-1 + (shape + 1) * r / (1 + r), 1 - shape * np.log1p(r)))
    else:
        z = shape * (np.log(y) - np.log(scale))
        value = np.log(shape) - np.log(y) + z - 2 * np.logaddexp(0, z)
        score = np.column_stack((shape * (2 * expit(z) - 1), 1 + z * (1 - 2 * expit(z))))
    return value, score


def objective(name, y, mass, design, offset, penalty):
    split = design.shape[1]

    def evaluate(beta):
        eta = np.column_stack((design @ beta[:split], design @ beta[split:])) + offset
        values, score = independent_rows(name, y, eta)
        value = -math.fsum(mass * values) + 0.5 * float(beta @ penalty @ beta)
        gradient = (
            -np.concatenate((design.T @ (mass * score[:, 0]), design.T @ (mass * score[:, 1])))
            + penalty @ beta
        )
        return value, gradient

    def score_error(beta):
        eta = np.column_stack((design @ beta[:split], design @ beta[split:])) + offset
        _, score = independent_rows(name, y, eta)
        # Outcome rounding model for these ordinary, conditioned fixtures:
        # O(n) dot-product accumulation and O(10) elementary row operations.
        # Include unit terms so cancellation of a row score is not its own
        # zero error estimate. No solver result enters this independent bound.
        magnitude = np.concatenate(
            [abs(design).T @ (mass * (1 + abs(score[:, j]))) for j in (0, 1)]
        ) + abs(penalty) @ abs(beta)
        u = 2.0**-53
        gamma = (len(y) + len(beta) + 32) * u / (1 - (len(y) + len(beta) + 32) * u)
        return 64 * gamma * magnitude

    evaluate.score_error = score_error
    return evaluate


def independent_curvature(evaluate, reference):
    count = len(reference)
    step = np.finfo(float).eps ** (1 / 3)
    return np.column_stack(
        [
            (
                evaluate(reference + step * np.eye(count)[j])[1]
                - evaluate(reference - step * np.eye(count)[j])[1]
            )
            / (2 * step)
            for j in range(count)
        ]
    )


def independent_covariance(evaluate, beta, actual):
    h = independent_curvature(evaluate, beta)
    h = (h + h.T) / 2
    expected = np.linalg.inv(h)
    # Central differences of the independent smooth gradient incur O(h^2)
    # truncation and O(u/h) rounding. Propagate that error through inversion
    # with ||A^-1||^2*||E||/(1-||A^-1 E||) on these conditioned fixtures.
    error_h = 128 * len(beta) * np.finfo(float).eps ** (2 / 3) * np.linalg.norm(h, 2)
    norm_inverse = np.linalg.norm(expected, 2)
    assert norm_inverse * error_h < 0.01
    bound = norm_inverse**2 * error_h / (1 - norm_inverse * error_h)
    assert np.linalg.norm(actual - expected, 2) <= bound
    return expected, bound


def assert_same_optimum(evaluate, candidate, reference, *, tolerance=1e-10):
    """Stationarity first, then forward comparisons on conditioned fixtures.

    The requested fit tolerance allows a relative score or squared Newton
    decrement <= tolerance*(1+|objective|). Both endpoints must independently
    meet that condition; a large residual cannot enlarge its acceptance limit.
    This is a local optimum comparison, not a global uniqueness claim, and
    must not be used for rank-deficient or weakly identified coefficient tests.
    """
    endpoints = []
    for beta in (candidate, reference):
        value, gradient = evaluate(beta)
        h = independent_curvature(evaluate, beta)
        h = (h + h.T) / 2
        eigen = np.linalg.eigvalsh(h)
        # Central differences incur O(u^(2/3)) relative curvature error.
        error_h = 128 * len(beta) * np.finfo(float).eps ** (2 / 3) * np.linalg.norm(h, 2)
        lower = eigen[0] - error_h
        assert lower > eigen[-1] * 1e-5, "fixture is not locally strongly convex"
        error_g = evaluate.score_error(beta)
        step = np.linalg.solve(h, gradient)
        # ||H^-1/2 e|| <= ||e||/sqrt(lambda_min); the curvature
        # perturbation inflates the squared norm by 1/(1-||E||/lambda_min).
        decrement = max(0.0, float(gradient @ step))
        decrement_upper = (np.sqrt(decrement) + np.linalg.norm(error_g) / np.sqrt(lower)) ** 2 / (
            1 - error_h / eigen[0]
        )
        limit = tolerance * (1 + abs(value))
        score_small = np.linalg.norm(gradient, np.inf) <= limit + np.linalg.norm(error_g, np.inf)
        assert score_small or decrement_upper <= limit, (
            f"nonstationary endpoint: score={np.linalg.norm(gradient):.6g}, "
            f"decrement={decrement_upper:.6g}, allowed={limit:.6g}"
        )
        endpoints.append((value, gradient, error_g, lower, eigen[-1] + error_h))
    (vc, gc, ec, lc, uc), (vr, gr, er, lr, ur) = endpoints
    residual = np.linalg.norm(gc) + np.linalg.norm(gr) + np.linalg.norm(ec) + np.linalg.norm(er)
    # These endpoints must lie in the same local strongly convex region.
    # The factor four allows its curvature variation around the endpoints.
    bound = 4 * residual / min(lc, lr) + 1024 * np.finfo(float).eps * (
        1 + np.linalg.norm(reference)
    )
    assert np.linalg.norm(candidate - reference) <= bound
    value_bound = residual * bound + max(uc, ur) * bound**2
    value_bound += 2048 * np.finfo(float).eps * (1 + abs(vc) + abs(vr))
    assert abs(vc - vr) <= value_bound


@pytest.mark.parametrize("cls,name", [(LomaxLSS, "lomax"), (LogLogisticLSS, "log_logistic")])
@pytest.mark.parametrize("mutation", ["unchanged_start", "displaced", "bad_reference"])
def test_optimum_oracle_rejects_nonstationary_endpoints(cls, name, mutation):
    family = cls()
    rng = np.random.default_rng(6201)
    y = family.quantile(rng.uniform(0.002, 0.998, 400), np.tile([2.0, 2.5], (400, 1)))
    evaluate = objective(
        name, y, np.ones(len(y)), np.ones((len(y), 1)), np.zeros((len(y), 2)), np.zeros((2, 2))
    )
    start = np.log([2.0, 2.0])
    reference = minimize(
        lambda b: evaluate(b)[0],
        start,
        jac=lambda b: evaluate(b)[1],
        method="BFGS",
        options={"gtol": 1e-10},
    ).x
    candidate = start if mutation == "unchanged_start" else reference + np.array([0.5, -0.5])
    if mutation == "bad_reference":
        # Even comparing an endpoint with itself must not establish an optimum.
        reference = candidate
    with pytest.raises(AssertionError):
        assert_same_optimum(evaluate, candidate, reference)


@pytest.mark.parametrize(
    "cls,name,scipy_law", [(LomaxLSS, "lomax", lomax), (LogLogisticLSS, "log_logistic", fisk)]
)
def test_intercept_mle_against_scipy_and_high_precision(cls, name, scipy_law):
    family = cls()
    rng = np.random.default_rng(6201)
    y = family.quantile(rng.uniform(0.002, 0.998, 400), np.tile([2.0, 2.5], (400, 1)))
    frame = pd.DataFrame({"unused": np.zeros(len(y))})
    model = SuperLSS(family, family.scale(), family.shape()).fit(frame, y, inner_tol=1e-10)
    candidate = np.log(model.predict_parameters(frame).to_numpy()[0])
    evaluate = objective(
        name, y, np.ones(len(y)), np.ones((len(y), 1)), np.zeros((len(y), 2)), np.zeros((2, 2))
    )
    reference = minimize(
        lambda b: evaluate(b)[0],
        np.log([2.0, 2.0]),
        jac=lambda b: evaluate(b)[1],
        method="BFGS",
        options={"gtol": 1e-10},
    )
    assert_same_optimum(evaluate, candidate, reference.x)
    scale, shape = np.exp(candidate)
    values = scipy_law(c=shape, loc=0, scale=scale).logpdf(y)
    np.testing.assert_allclose(math.fsum(values), -evaluate(candidate)[0], rtol=512 * 2.0**-53)
    with mp.workdps(70):
        oracle = mp.fsum(
            log_density(name, mp.mpf(float(z)), mp.mpf(scale), mp.mpf(shape)) for z in y
        )
    assert abs(float(oracle) - math.fsum(values)) <= 1024 * 2.0**-53 * max(1.0, float(abs(oracle)))


@pytest.mark.parametrize("cls,name", [(LomaxLSS, "lomax"), (LogLogisticLSS, "log_logistic")])
def test_fixed_penalty_independent_objective_chunk_replay_and_offsets(cls, name):
    family = cls()
    rng = np.random.default_rng(6202)
    x = np.linspace(-1, 1, 240)
    frame = pd.DataFrame({"x": x})
    offset = np.column_stack((0.1 * np.sin(2 * x), 0.05 * np.cos(x)))
    eta = np.column_stack((0.4 + 0.3 * x, 0.8 - 0.15 * x)) + offset
    y = family.quantile(rng.uniform(0.002, 0.998, len(x)), np.exp(eta))
    mass = rng.integers(1, 4, len(x))
    resolved = weights(mass)
    layout = build_stacked_layout(
        compile_predictors(
            as_eager_frame(frame),
            resolved,
            family.parameters,
            (Predictor("scale", {"x": Numeric()}), Predictor("shape", {"x": Numeric()})),
            offsets={"scale": offset[:, 0], "shape": offset[:, 1]},
        )
    )
    plan = family.bind_likelihood(y, resolved, COMPLETE_OBSERVATION)
    penalty = np.diag([0.0, 2.0, 0.0, 3.0])
    config = DenseSolverConfig(tolerance=1e-10)
    dense = fit_dense_fixed_lambda(family, layout, y, plan, penalty, config=config, chunk_size=None)
    chunk = fit_dense_fixed_lambda(family, layout, y, plan, penalty, config=config, chunk_size=37)
    assert dense.converged and chunk.converged
    design = np.column_stack((np.ones(len(x)), x))
    evaluate = objective(name, y, mass, design, offset, penalty)
    ref = minimize(
        lambda b: evaluate(b)[0],
        np.array([0.4, 0.3, 0.8, -0.15]),
        jac=lambda b: evaluate(b)[1],
        method="BFGS",
        options={"gtol": 1e-9},
    )
    assert_same_optimum(evaluate, dense.coefficients, ref.x)
    assert_same_optimum(evaluate, chunk.coefficients, ref.x)
    np.testing.assert_allclose(
        dense.terminal_penalized_curvature, chunk.terminal_penalized_curvature, rtol=1e-8, atol=1e-9
    )
    assert chunk.resolved_chunk_size == 37
    assert chunk.terminal_rank.rank == dense.terminal_rank.rank == 4
    assert chunk.execution_backend_identifier != dense.execution_backend_identifier


def test_lomax_gpd_overlap_likelihood_derivatives_mle_and_covariance():
    f, g = LomaxLSS(), GeneralizedParetoLSS()
    theta = np.array([[2.0, 1.5], [0.5, 3.0], [10.0, 25.0]])
    transformed = np.column_stack((theta[:, 0] / theta[:, 1], 1 / theta[:, 1]))
    y = np.array([0.0, 0.5, 4.0])
    w = weights([1, 2, 5])
    p = f.bind_likelihood(y, w, COMPLETE_OBSERVATION)
    q = g.bind_likelihood(y, w, COMPLETE_OBSERVATION)
    a = f.evaluate_natural(y, theta, p)
    b = g.evaluate_natural(y, transformed, q)
    np.testing.assert_allclose(
        a.reported_log_likelihood,
        b.reported_log_likelihood,
        rtol=1024 * 2.0**-53,
        atol=1024 * 2.0**-53,
    )
    for i, (scale, shape) in enumerate(theta):
        jac = np.array([[1 / shape, -scale / shape**2], [0.0, -1 / shape**2]])
        np.testing.assert_allclose(
            a.score[i], jac.T @ b.score[i], rtol=1024 * 2.0**-53, atol=1024 * 2.0**-53
        )
        h = np.array(
            [
                [b.hessian_packed[i, 0], b.hessian_packed[i, 1]],
                [b.hessian_packed[i, 1], b.hessian_packed[i, 2]],
            ]
        )
        expected = jac.T @ h @ jac + b.score[i, 0] * np.array(
            [[0.0, -1 / shape**2], [-1 / shape**2, 2 * scale / shape**3]]
        )
        expected[1, 1] += b.score[i, 1] * 2 / shape**3
        np.testing.assert_allclose(
            a.hessian_packed[i],
            expected[[0, 0, 1], [0, 1, 1]],
            rtol=4096 * 2.0**-53,
            atol=4096 * 2.0**-53,
        )
    for lo, hi in ((0.0, 3.0), (0.1, np.inf), (0.2, 2.0)):
        np.testing.assert_allclose(
            f.log_window_probability(lo, hi, theta),
            g.log_window_probability(lo, hi, transformed),
            rtol=4096 * 2.0**-53,
            atol=4096 * 2.0**-53,
        )
    rng = np.random.default_rng(6203)
    y = genpareto(c=0.4, scale=0.8).rvs(500, random_state=rng)
    frame = pd.DataFrame({"unused": np.zeros(len(y))})
    fm = SuperLSS(f, f.scale(), f.shape()).fit(frame, y, inner_tol=1e-10)
    gm = SuperLSS(g, g.scale(), g.shape()).fit(frame, y, inner_tol=1e-10)
    ft = fm.predict_parameters(frame).to_numpy()[0]
    gt = gm.predict_parameters(frame).to_numpy()[0]
    evaluate = objective(
        "lomax", y, np.ones(len(y)), np.ones((len(y), 1)), np.zeros((len(y), 2)), np.zeros((2, 2))
    )
    g_as_lomax = np.log([gt[0] / gt[1], 1 / gt[1]])
    assert_same_optimum(evaluate, np.log(ft), g_as_lomax)
    # Transform log-Lomax and GPD (log scale, logit xi) covariance to
    # natural GPD coordinates before comparing the estimable intercept MLE.
    cf = fm.covariance_
    cg = gm.covariance_
    jf = np.array([[ft[0] / ft[1], -ft[0] / ft[1]], [0.0, -1 / ft[1]]])
    jg = np.diag([gt[0], gt[1] * (1 - gt[1])])
    ref_f, error_f = independent_covariance(evaluate, np.log(ft), cf)

    def gpd_objective(beta):
        xi = expit(beta[1])
        lomax_beta = np.array([beta[0] - np.log(xi), -np.log(xi)])
        value, gradient = evaluate(lomax_beta)
        jac = np.array([[1.0, -(1 - xi)], [0.0, -(1 - xi)]])
        return value, jac.T @ gradient

    ref_g, error_g = independent_covariance(
        gpd_objective, np.array([np.log(gt[0]), np.log(gt[1] / (1 - gt[1]))]), cg
    )
    actual_difference = jf @ cf @ jf.T - jg @ cg @ jg.T
    # The independently evaluated curvature accounts for the small difference
    # in fitted parameter states and the residual score's coordinate term.
    reference_difference = jf @ ref_f @ jf.T - jg @ ref_g @ jg.T
    bound = np.linalg.norm(jf, 2) ** 2 * error_f + np.linalg.norm(jg, 2) ** 2 * error_g
    assert np.linalg.norm(actual_difference - reference_difference, 2) <= bound


@pytest.mark.parametrize("cls", [LomaxLSS, LogLogisticLSS])
def test_smooth_reml_starts_backends_compaction_and_serialization(cls, record_property):
    import json

    family = cls()
    rng = np.random.default_rng(6204)
    x = np.tile(np.linspace(-1, 1, 32), 8)
    frame = pd.DataFrame({"x": x})
    theta = np.column_stack((np.exp(0.3 + 0.25 * np.sin(2 * x)), np.exp(0.9 + 0.1 * x)))
    y = family.quantile(rng.uniform(0.002, 0.998, len(x)), theta)
    outputs = []
    for outer, start in product((None, "efs", "efs+newton"), (0.1, 1.0, 10.0)):
        model = SuperLSS(
            family,
            family.scale(s("x", k=4)),
            family.shape(s("x", k=4)),
            discrete=start == 10.0,
            n_bins=64,
        )
        model.fit_reml(
            frame,
            y,
            outer=outer,
            initial_lambda=start,
            max_reml_iter=100,
            retain_rows=start != 10.0,
        )
        result = model.result_
        assert np.isfinite(model.predict_quantile(frame, 0.9)).all()
        assert np.isfinite(model.covariance_).all()
        assert model.smoothing_convergence_reason_
        assert isinstance(model.smoothing_certified_, bool)
        diagnostic = model.diagnose().to_dict()
        assert diagnostic
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(
            restored.predict_parameters(frame), model.predict_parameters(frame)
        )
        fitted = model._require_fitted()
        smoothing = fitted.smoothing
        assert model.smoothing_certified_ == smoothing.matched_certified
        if model.smoothing_certified_:
            assert smoothing.terminal_evidence_fresh and fitted.result.converged
        if smoothing.convergence_reason in ("max_iterations", "practical_plateau"):
            assert not model.smoothing_certified_
        outputs.append(
            dict(
                outer=outer,
                start=start,
                log_likelihood=float(fitted.result.log_likelihood),
                objective=smoothing.objective,
                certified=model.smoothing_certified_,
                reason=str(model.smoothing_convergence_reason_),
                rank=fitted.result.terminal_rank.rank,
                edf=float(np.trace(fitted.covariance @ fitted.result.terminal_data_curvature)),
                covariance_diagonal=np.diag(model.covariance_).tolist(),
                q90=model.predict_quantile(frame.iloc[:4], 0.9).tolist(),
                backend=str(fitted.result.execution_backend_identifier),
                warnings=list(result.warnings),
                gradient_certificate=dict(smoothing.terminal_gradient_certificate or {}),
                projected_gradient=smoothing.terminal_projected_gradient_norm,
            )
        )
    # Different outer algorithms may stop at different smoothing solutions.
    # Their disclosed states are retained; no assertion invents certification.
    assert len(outputs) == 9
    reference = outputs[1]
    for row in outputs:
        row["comparison_to_default_start_one"] = {
            "objective_difference": row["objective"] - reference["objective"],
            "edf_difference": row["edf"] - reference["edf"],
            "rank_difference": row["rank"] - reference["rank"],
            "maximum_q90_difference": float(
                np.max(np.abs(np.array(row["q90"]) - reference["q90"]))
            ),
            "maximum_covariance_diagonal_difference": float(
                np.max(
                    np.abs(np.array(row["covariance_diagonal"]) - reference["covariance_diagonal"])
                )
            ),
        }
    record_property("reml_endpoints", json.dumps(outputs))


def test_complete_fit_benchmark_has_reviewable_receipt(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).parents[1] / "benchmarks/declared_scalar_families.py"
    assert script.exists(), "complete-fit comparison harness is missing"
    output = tmp_path / "receipt.json"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--case",
            "lomax-intercept",
            "--rows",
            "160",
            "--repeats",
            "1",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())
    assert report["status"] == "completed"
    assert report["baseline_sha"] == "6643ed9c94497eba0efb8869b945a17c5d380caf"
    cases = report["cases"]
    assert len(cases) == 2
    for case in cases:
        assert case["status"] == "completed", case
        assert case["actual_backend"] and case["generated_dispatch"]["density"]["rows"] > 0
        assert case["memory"]["peak_rss_bytes"] > 0 and case["memory"]["retained_array_bytes"] > 0
        assert case["validation"]["heldout_seed"] != case["train_seed"]
    np.testing.assert_allclose(cases[0]["parameters"], cases[1]["parameters"], rtol=1e-7, atol=0)
