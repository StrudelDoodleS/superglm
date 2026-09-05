"""Regression tests for the canonical Binomial REML gradient path."""

import math

import numpy as np
import pandas as pd

from superglm import SuperGLM
from superglm.distributions import Binomial
from superglm.features import Categorical, Spline
from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
from superglm.links import LogitLink, stabilize_eta
from superglm.reml.observed_geometry import ObservedREMLGeometry
from superglm.reml.w_derivatives import reml_w_correction
from superglm.solvers.pirls import PIRLSResult, REMLGeometrySummary
from superglm.types import GroupSlice, PenaltyComponent


def _four_smooth_fit(*, link: str = "logit") -> SuperGLM:
    rng = np.random.default_rng(9401)
    n = 600
    x0 = rng.uniform(-1.0, 1.0, n)
    x1 = rng.uniform(-1.0, 1.0, n)
    x2 = rng.uniform(-1.0, 1.0, n)
    x3 = rng.uniform(-1.0, 1.0, n)
    offset = 0.12 * np.cos(1.3 * x1)
    eta = -0.2 + 0.8 * np.sin(2.0 * x0) - 0.5 * x1 + 0.25 * x2**2 - 0.15 * x3 + offset
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(np.float64)
    weights = rng.integers(1, 4, n).astype(np.float64)
    frame = pd.DataFrame({"x0": x0, "x1": x1, "x2": x2, "x3": x3})

    return SuperGLM(
        family="binomial",
        link=link,
        weight_semantics="frequency",
        selection_penalty=0.0,
        features={
            "x0": Spline(n_knots=6, penalty="ssp"),
            "x1": Spline(n_knots=6, penalty="ssp"),
            "x2": Spline(n_knots=6, penalty="ssp"),
            "x3": Spline(n_knots=6, penalty="ssp"),
        },
    ).fit_reml(
        frame,
        y,
        sample_weight=weights,
        offset=offset,
        runtime_validation="skip",
    )


def _correction_case(*, zero_weight_outlier: bool = False):
    rng = np.random.default_rng(9403)
    n = 80
    columns = rng.normal(size=(n, 4))
    sample_weight = np.ones(n)
    if zero_weight_outlier:
        columns[0, 0] = 1.0e200
        sample_weight[0] = 0.0
    beta = np.array([0.12, -0.08, 0.06, 0.04])
    intercept = -0.35
    link = LogitLink()
    distribution = Binomial()
    eta = stabilize_eta(columns @ beta + intercept, link)
    mu = link.inverse(eta)
    working_weight = sample_weight * mu * (1.0 - mu)
    active = working_weight > 0.0
    sum_w = float(np.sum(working_weight[active]))
    mean_x = (
        np.sum(
            working_weight[active, None] * columns[active],
            axis=0,
        )
        / sum_w
    )
    centered = columns[active] - mean_x
    data_gram = centered.T @ (working_weight[active, None] * centered)
    lambdas = {f"x{i}": float(i + 1) for i in range(4)}
    inverse = np.linalg.inv(data_gram + np.diag(list(lambdas.values())))
    groups = [GroupSlice(name=f"x{i}", start=i, end=i + 1) for i in range(4)]
    penalties = [
        PenaltyComponent(
            name=group.name,
            group_name=group.name,
            group_index=i,
            group_sl=group.sl,
            omega_raw=None,
            penalty_kind="identity",
        )
        for i, group in enumerate(groups)
    ]
    dm = DesignMatrix(
        [DenseGroupMatrix(columns[:, i : i + 1]) for i in range(4)],
        n=n,
        p=4,
    )
    result = PIRLSResult(
        beta=beta,
        intercept=intercept,
        n_iter=1,
        deviance=0.0,
        converged=True,
        phi=1.0,
        effective_df=4.0,
        reml_geometry=REMLGeometrySummary(
            mean_x=mean_x,
            sum_w=sum_w,
            column_scale=np.sqrt(np.diag(data_gram)),
        ),
    )
    common = {
        "dm": dm,
        "link": link,
        "groups": groups,
        "pirls_result": result,
        "XtWX_S_inv": inverse,
        "lambdas": lambdas,
        "sample_weight": sample_weight,
        "offset_arr": np.zeros(n),
        "distribution": distribution,
        "reml_penalties": penalties,
    }
    return common


def _translated_geometry_case(translation: float):
    common = _correction_case()
    original = common["dm"].toarray()
    translated = original + translation
    result = common["pirls_result"]
    result.intercept -= translation * float(np.sum(result.beta))
    eta = stabilize_eta(translated @ result.beta + result.intercept, common["link"])
    mu = common["link"].inverse(eta)
    working_weight = mu * (1.0 - mu)
    sum_w = float(np.sum(working_weight))
    anchor = translated[0]
    differences = translated - anchor
    mean_difference = np.sum(working_weight[:, None] * differences, axis=0) / sum_w
    mean_x = anchor + mean_difference
    centered = differences - mean_difference
    centered_data_gram = centered.T @ (working_weight[:, None] * centered)
    centered_hessian = centered_data_gram + np.diag(list(common["lambdas"].values()))
    inverse = np.linalg.inv(centered_hessian)
    common["XtWX_S_inv"] = inverse
    common["dm"] = DesignMatrix(
        [DenseGroupMatrix(translated[:, i : i + 1]) for i in range(4)],
        n=len(translated),
        p=4,
    )
    common["geometry"] = ObservedREMLGeometry(
        eta=eta,
        mu=mu,
        weights=working_weight,
        weight_derivative=working_weight * (1.0 - 2.0 * mu),
        weight_second_derivative=None,
        sum_w=sum_w,
        mean_x=mean_x,
        centered_data_gram=centered_data_gram,
        centered_hessian=centered_hessian,
        hessian_inverse=inverse,
        log_det_H=float(np.log(sum_w) + np.linalg.slogdet(centered_hessian)[1]),
        hessian_rank=1 + int(np.linalg.matrix_rank(centered_hessian)),
    )
    return common


def _scalar_centered_correction(common) -> np.ndarray:
    """Reference the leverage identity with scalar compensated reductions."""
    geometry = common["geometry"]
    centered = common["dm"].toarray() - geometry.mean_x
    inverse = common["XtWX_S_inv"]
    leverage = np.array(
        [
            math.fsum(
                float(centered[i, j] * inverse[j, k] * centered[i, k])
                for j in range(4)
                for k in range(4)
            )
            for i in range(len(centered))
        ]
    )
    row_channel = 0.5 * geometry.weight_derivative * (leverage + 1.0 / geometry.sum_w)
    rhs = np.array(
        [
            math.fsum(float(centered[i, j] * row_channel[i]) for i in range(len(centered)))
            for j in range(4)
        ]
    )
    penalty_rhs = np.diag(np.array(list(common["lambdas"].values())) * common["pirls_result"].beta)
    directions = -inverse @ penalty_rhs
    return np.array(
        [math.fsum(float(rhs[k] * directions[k, j]) for k in range(4)) for j in range(4)]
    )


def test_binomial_logit_reml_uses_one_leverage_gradient_pass() -> None:
    """Reverting to one derivative Gram per smooth must fail this dispatch gate."""
    model = _four_smooth_fit()

    assert model._reml_profile["reml_w_correction_mode"] == "leverage_gradient"


def test_profile_reports_the_correction_geometry_that_actually_ran(monkeypatch) -> None:
    """Ignoring the fast-path request must not leave a false leverage label."""
    import superglm.reml.direct as direct

    original = direct.reml_w_correction

    def force_derivative_grams(*args, **kwargs):
        kwargs["gradient_only"] = False
        return original(*args, **kwargs)

    monkeypatch.setattr(direct, "reml_w_correction", force_derivative_grams)

    model = _four_smooth_fit()

    assert model._reml_profile["reml_w_correction_mode"] == "derivative_grams"


def test_leverage_gradient_matches_the_full_derivative_gram_result() -> None:
    """Dropping centering or the intercept determinant term must change this result."""
    from superglm.reml.w_derivatives import reml_w_correction

    model = _four_smooth_fit()
    inverse = np.asarray(model._fit_inference_info["XtWX_inv"], dtype=np.float64)
    kwargs = {
        "sample_weight": np.asarray(model._fit_weights, dtype=np.float64),
        "offset_arr": np.asarray(model._fit_offset, dtype=np.float64),
        "distribution": model._distribution,
        "reml_penalties": model._reml_penalties,
    }
    full = reml_w_correction(
        model._dm,
        model._link,
        model._groups,
        model.result,
        inverse,
        model._reml_lambdas,
        gradient_only=False,
        **kwargs,
    )
    leverage = reml_w_correction(
        model._dm,
        model._link,
        model._groups,
        model.result,
        inverse,
        model._reml_lambdas,
        gradient_only=True,
        **kwargs,
    )

    assert full is not None
    assert leverage is not None
    scale = max(float(np.linalg.norm(full[0], ord=np.inf)), 1.0)
    tolerance = 512.0 * np.finfo(np.float64).eps * max(model._dm.p, 1) * scale
    np.testing.assert_allclose(leverage[0], full[0], rtol=0.0, atol=tolerance)
    assert leverage[1] is None


def test_leverage_gradient_ignores_zero_weight_outlier_geometry() -> None:
    """An extreme row with exactly zero derivative weight must not create NaNs."""
    common = _correction_case(zero_weight_outlier=True)

    full = reml_w_correction(**common, gradient_only=False)
    leverage = reml_w_correction(**common, gradient_only=True)

    assert full is not None
    assert leverage is not None
    assert np.all(np.isfinite(full[0]))
    np.testing.assert_allclose(leverage[0], full[0], rtol=2e-13, atol=2e-15)


def test_leverage_gradient_is_stable_for_translated_columns() -> None:
    """Raw transpose cancellation must not corrupt an otherwise centered result."""
    translation = 1.0e10
    common = _translated_geometry_case(translation)

    leverage = reml_w_correction(**common, gradient_only=True)

    assert leverage is not None
    expected = _scalar_centered_correction(common)
    scale = max(float(np.linalg.norm(expected, ord=np.inf)), 1.0)
    tolerance = 512.0 * np.finfo(np.float64).eps * 4.0 * scale
    np.testing.assert_allclose(leverage[0], expected, rtol=0.0, atol=tolerance)


def test_one_smooth_with_many_categories_keeps_specialized_derivative_gram() -> None:
    """A dense leverage pass must not replace the cheaper sparse one-penalty path."""
    rng = np.random.default_rng(9402)
    n = 1_200
    x = rng.uniform(-1.0, 1.0, n)
    segment = rng.integers(0, 60, n).astype(str)
    eta = -0.3 + 0.7 * np.sin(2.0 * x) + 0.08 * (segment.astype(int) % 5)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(np.float64)
    frame = pd.DataFrame({"x": x, "segment": segment})

    model = SuperGLM(
        family="binomial",
        selection_penalty=0.0,
        features={
            "x": Spline(n_knots=6, penalty="ssp"),
            "segment": Categorical(base="first"),
        },
    ).fit_reml(frame, y, runtime_validation="skip")

    assert model._reml_profile["reml_w_correction_mode"] == "derivative_grams"


def test_noncanonical_binomial_keeps_full_derivative_geometry() -> None:
    """The leverage shortcut is deliberately narrower than all Binomial fits."""
    model = _four_smooth_fit(link="probit")

    assert model._reml_profile["reml_w_correction_mode"] == "derivative_grams"


def test_profile_reports_when_weight_correction_is_structurally_zero() -> None:
    """Preselecting a fast route must not claim work that never ran."""
    rng = np.random.default_rng(9404)
    unique_rows = 120
    frame = pd.DataFrame(
        {f"x{i}": np.repeat(rng.uniform(-1.0, 1.0, unique_rows), 2) for i in range(4)}
    )
    y = np.tile(np.array([0.0, 1.0]), unique_rows)

    model = SuperGLM(
        family="binomial",
        selection_penalty=0.0,
        features={f"x{i}": Spline(n_knots=6, penalty="ssp") for i in range(4)},
    ).fit_reml(frame, y, runtime_validation="skip")

    assert model._reml_profile["reml_w_correction_mode"] == "none"
