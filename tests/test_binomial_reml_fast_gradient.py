"""Binomial/logit REML takes the W(rho) leverage route every family takes above its work bar.

On a dense factor binomial/logit takes it below the bar too: a design of
lookups reads its leverages from term-pair tables
(``_group_matrix/_lookup_tables.py``), any other design solves its rows
(``reml/_leverage_rows.py``).  On either, its Newton model and its stop's
second-order check leave out the W(rho) cross terms, as the pass it took
before did.  Pinned here on binomial/logit, whose weight derivative
``W (1 - 2 mu)`` changes sign across the rows.
"""

import math

import numpy as np
import pandas as pd
import pytest

import superglm._group_matrix._lookup_tables as leverage_tables
import superglm.reml.direct as direct
from superglm import SuperGLM
from superglm.distributions import Binomial
from superglm.features import Categorical, Numeric, Spline
from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
from superglm.links import LogitLink, stabilize_eta
from superglm.reml.observed_geometry import ObservedREMLGeometry
from superglm.reml.w_derivatives import reml_w_correction
from superglm.solvers.pirls import PIRLSResult, REMLGeometrySummary
from superglm.types import GroupSlice, PenaltyComponent
from tests.test_reml_leverage_hessian import (
    _correction_inputs,
    _penalty_operands,
    _poisson_frame,
    _rounding_bounds,
)
from tests.test_reml_leverage_tables import _count_roundings, _force_candidate_stops


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


def _lookup_fit() -> SuperGLM:
    """Binomial/logit on coarse covariates: every group a lookup or a dense column."""
    frame, _, _ = _poisson_frame(1500, 1.0e3)
    eta = -0.3 + np.sin(3.0 * frame.a) - 0.2 * frame.b + 0.1 * (frame.c.astype(int) % 3)
    rng = np.random.default_rng(9405)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(np.float64)
    return SuperGLM(
        family="binomial",
        features={
            "a": Spline(n_knots=6),
            "b": Spline(n_knots=6),
            "e": Spline(n_knots=5),
            "c": Categorical(),
            "d": Numeric(),
        },
        interactions=[("a", "b"), ("e", "c")],
        discrete=False,
    ).fit_reml(frame, y, runtime_validation="skip")


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
    """Reference the leverage identity with scalar compensated reductions.

    Rows are centred as the fixture's geometry centres them, ``(x - x_0) -
    d`` with ``d`` the compensated mean difference: about the rounded
    ``mean_x`` the reference itself would carry ``u |translation|`` into
    every row (issue #430).
    """
    geometry = common["geometry"]
    design = common["dm"].toarray()
    differences = design - design[0]
    weights = np.asarray(geometry.weights, dtype=np.float64)
    mean_difference = np.array(
        [math.fsum(weights * differences[:, j]) for j in range(design.shape[1])]
    ) / float(geometry.sum_w)
    centered = differences - mean_difference
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


def test_binomial_logit_reml_reads_its_leverages_from_the_tables() -> None:
    """A lookup design takes the table route, far below the leverage work bar.

    Binomial/logit took a dense row pass of its own there, ahead of the table
    route every other family takes; restoring that gate fails this test.
    """
    model = _lookup_fit()

    assert model._reml_profile["reml_w_correction_mode"] == "leverage_tables"
    assert model._reml_profile["reml_n_w_correction"] > 0


def test_profile_reports_the_correction_geometry_that_actually_ran(monkeypatch) -> None:
    """Ignoring the leverage request must not leave a false leverage label."""
    original = direct.reml_w_correction

    def force_derivative_grams(*args, **kwargs):
        kwargs["leverage_hessian"] = False
        return original(*args, **kwargs)

    monkeypatch.setattr(direct, "reml_w_correction", force_derivative_grams)

    model = _lookup_fit()

    assert model._reml_profile["reml_w_correction_mode"] == "derivative_grams"


def test_binomial_logit_table_gradient_matches_the_derivative_grams_within_rounding() -> None:
    """The table route's gradient is the derivative Grams' exact one, within their rounding.

    Bound: ``_rounding_bounds`` in ``u``, the table route's row sums counted by
    ``_count_roundings`` (the leverage bound ``gamma_K eta_r`` of
    ``_lookup_tables``); it is below a millionth of the terms, so a dropped or
    mis-weighted term fails it.  The weight derivative ``W (1 - 2 mu)`` is not
    proportional to ``W``, so the profiled intercept's ``1 / sum W`` term does
    not cancel against the centred rows as it does for Poisson/log: dropping
    it from ``_leverage_table_gradient`` fails here.
    """
    model = _lookup_fit()
    plan = leverage_tables.lookup_plan(model._dm)
    assert plan is not None
    gradient, _, gradient_bound, _ = _rounding_bounds(
        model, raw_centre_copies=2, extra_roundings=_count_roundings(plan)
    )
    args, kwargs = _correction_inputs(model)

    tables = reml_w_correction(*args, leverage_hessian=True, **kwargs)

    assert tables is not None and tables[1] is None and tables[2] is None
    assert np.all(np.abs(tables[0] - gradient) <= gradient_bound)
    assert np.max(gradient_bound) < 1e-6 * np.max(np.abs(gradient))


def test_leverage_route_ignores_zero_weight_outlier_geometry() -> None:
    """An extreme row with exactly zero derivative weight must not create NaNs."""
    common = _correction_case(zero_weight_outlier=True)

    full = reml_w_correction(**common)
    leverage = reml_w_correction(**common, leverage_hessian=True)

    assert full is not None and full[1] is not None
    # the leverage route ran: it forms no derivative Gram (a declined request returns them)
    assert leverage is not None and len(leverage) == 3 and leverage[1] is None
    assert np.all(np.isfinite(full[0]))
    np.testing.assert_allclose(leverage[0], full[0], rtol=2e-13, atol=2e-15)


def test_leverage_route_is_stable_for_translated_columns() -> None:
    """Raw transpose cancellation must not corrupt an otherwise centered result."""
    translation = 1.0e10
    common = _translated_geometry_case(translation)

    leverage = reml_w_correction(**common, leverage_hessian=True)

    assert leverage is not None and len(leverage) == 3 and leverage[1] is None
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


@pytest.mark.parametrize(
    ("link", "mode"), [("logit", "leverage_tables"), ("probit", "derivative_grams")]
)
def test_below_the_leverage_work_bar_only_the_logit_link_keeps_one_pass(link, mode) -> None:
    """Four continuous smooths on 600 rows, far below the bar and no lookups.

    Binomial/logit solves its rows and keeps the gradient alone in its Newton
    model (``leverage_tables`` labels the leverage route's gradient-only model;
    there is no table here); probit, like every other pair, keeps the
    derivative Grams.  Sending binomial/logit to the bar fails the logit case,
    and so does adding the W(rho) cross terms to its Newton model (the label
    is then ``leverage_hessian``).
    """
    model = _four_smooth_fit(link=link)

    assert leverage_tables.lookup_plan(model._dm) is None
    assert model._reml_profile["reml_w_correction_mode"] == mode


@pytest.mark.parametrize("design", ["lookups", "continuous"])
def test_a_binomial_logit_stop_reads_the_hessian_of_its_newton_model(monkeypatch, design) -> None:
    """Every iteration's correction is the gradient alone, and a candidate stop solves no row.

    Binomial/logit keeps the Newton model and the stop of the pass it took
    before: both read the fixed-W outer Hessian, without the W(rho) cross
    terms, on the lookup design and on the continuous one.  The stops are
    forced (``_force_candidate_stops``), so the check does not wait on the
    fit's own exit.  Mutations: a stop pass for binomial/logit (``table_route``)
    fails both cases; the cross terms in the solved-row Newton model
    (``gradient_terms_only`` for structured models alone) fails the
    continuous one.
    """
    _force_candidate_stops(monkeypatch)
    correction, hessian = direct.reml_w_correction, direct.reml_direct_hessian
    concave = direct.concave_live_directions
    corrections, hessians, reads = [], [], []

    def recorded_correction(*args, **kwargs):
        corrections.append((kwargs.get("lookup_tables", True), correction(*args, **kwargs)))
        return corrections[-1][1]

    def recorded_hessian(*args, **kwargs):
        hessians.append(hessian(*args, **kwargs))
        return hessians[-1]

    def read(gradient, hess, estimated, **kwargs):
        reads.append((np.array(hess, copy=True), hessians[-1]))
        return concave(gradient, hess, estimated, **kwargs)

    monkeypatch.setattr(direct, "reml_w_correction", recorded_correction)
    monkeypatch.setattr(direct, "reml_direct_hessian", recorded_hessian)
    monkeypatch.setattr(direct, "concave_live_directions", read)

    model = _lookup_fit() if design == "lookups" else _four_smooth_fit(link="logit")

    assert (leverage_tables.lookup_plan(model._dm) is not None) == (design == "lookups")
    assert corrections and reads
    for lookup_tables, result in corrections:
        assert lookup_tables  # no solved-row pass for a stop
        assert result is not None and len(result) == 3
        assert result[1] is None and result[2] is None
    for hess, fixed_w in reads:
        assert np.array_equal(hess, fixed_w)
    assert model._reml_profile["reml_n_stop_cross_passes"] == 0


def _fit_and_outer_hessian(monkeypatch, fit, *, grams: bool):
    """``fit()``, on the derivative Grams when ``grams``, and the last outer Hessian it formed."""
    correction, hessian = direct.reml_w_correction, direct.reml_direct_hessian
    last = {}

    def routed(*args, **kwargs):
        if grams:
            kwargs["leverage_hessian"] = False
        return correction(*args, **kwargs)

    def recorded(*args, **kwargs):
        last["hessian"] = hessian(*args, **kwargs)
        return last["hessian"]

    with monkeypatch.context() as patch:
        patch.setattr(direct, "reml_w_correction", routed)
        patch.setattr(direct, "reml_direct_hessian", recorded)
        model = fit()
    return model, last["hessian"]


def _frozen(model) -> np.ndarray:
    decision = model._reml_profile.get("reml_freeze_decision") or {}
    return np.asarray(decision.get("frozen", np.zeros(len(model._reml_penalties), dtype=bool)))


def _fit_sensitivities(model) -> tuple[np.ndarray, np.ndarray]:
    """``d eta / d rho`` on the fitted rows and ``d tr(H^-1 A) / d rho`` at a binomial/logit fit.

    By the implicit function theorem ``dbeta_j = -H^-1 S_j beta`` and ``J_j =
    X_c dbeta_j``.  With ``A = X_c' W X_c``, ``S`` the summed penalties and
    ``C_j = X_c' diag(W' J_j) X_c``, which is ``dA/drho_j`` exactly to first
    order (the weighted means move with ``W``), ``tr(H^-1 A)`` moves by
    ``tr(H^-1 C_j H^-1 S) - tr(H^-1 S_j H^-1 A)``.
    """
    inverse = np.asarray(model._fit_inference_info["XtWX_inv"], dtype=np.float64)
    X = model._dm.toarray()
    centred = X - np.asarray(model.result.reml_geometry.mean_x)
    offset = 0.0 if model._fit_offset is None else np.asarray(model._fit_offset)
    mu = 1.0 / (1.0 + np.exp(-(X @ model.result.beta + model.result.intercept + offset)))
    weights = np.asarray(model._fit_weights) * mu * (1.0 - mu)
    smoothing, penalty_rhs = _penalty_operands(
        model._dm, model._reml_penalties, model._reml_lambdas, model.result.beta
    )
    J = centred @ (-inverse @ penalty_rhs)
    data = inverse @ (centred.T @ (weights[:, None] * centred))
    penalty = inverse @ smoothing.sum(axis=2)
    derivative = weights * (1.0 - 2.0 * mu)
    edf = np.array(
        [
            np.trace(inverse @ (centred.T @ ((derivative * J[:, j])[:, None] * centred)) @ penalty)
            - np.trace(inverse @ smoothing[:, :, j] @ data)
            for j in range(J.shape[1])
        ]
    )
    return J, edf


@pytest.mark.parametrize("design", ["lookups", "continuous"])
def test_a_binomial_logit_fit_lands_where_the_derivative_grams_land(monkeypatch, design) -> None:
    """The leverage route's fit is the derivative Grams' fit, to what the REML stop certifies.

    The two routes form one exact gradient of one objective; the Grams' Newton
    model also carries the first-order weight terms, so the iterates differ and
    each fit stops on the compound test, the projected gradient and the
    objective change below ``tau (1 + |V|)``, ``tau = reml_tol``.  Hence

    - the objectives agree to ``tau (1 + |V|)`` (tests/test_nested_structured_fit.py);
    - both stops freeze the same directions;
    - on every other direction the gradients differ by under ``2 tau (1 + |V|)``,
      to first order ``H delta`` with ``H`` the outer Hessian at the Grams'
      published point, so ``|delta| <= |H^-1| 2 tau (1 + |V|)`` for the log
      smoothing parameters;
    - a frozen direction is at working infinity, where no stop test resolves
      its smoothing parameter (Wood, Pya & Saefken 2016, sec. 3.1); it is
      compared through the fit it moves, its own ``|delta|`` plus the
      second-order remainder of an exponential approach, ``|delta|^2 / 2``;
    - the linear predictor on the fitted rows and the EDF move by the
      first-order images of those steps (``_fit_sensitivities``).

    Mutation: the leverage route's gradient correction halved (its stationary
    point moves).
    """
    fit = _lookup_fit if design == "lookups" else lambda: _four_smooth_fit(link="logit")
    leverage, _ = _fit_and_outer_hessian(monkeypatch, fit, grams=False)
    grams, hessian = _fit_and_outer_hessian(monkeypatch, fit, grams=True)
    assert leverage._reml_profile["reml_w_correction_mode"] == "leverage_tables"
    assert grams._reml_profile["reml_w_correction_mode"] == "derivative_grams"
    assert leverage._reml_result.converged and grams._reml_result.converged
    objective = grams._reml_result.objective
    bar = grams._reml_profile["reml_tol_resolved"] * (1.0 + abs(objective))
    assert abs(leverage._reml_result.objective - objective) <= bar

    names = [pc.name for pc in grams._reml_penalties]
    delta = np.log([leverage._reml_lambdas[name] for name in names]) - np.log(
        [grams._reml_lambdas[name] for name in names]
    )
    frozen = _frozen(grams)
    np.testing.assert_array_equal(_frozen(leverage), frozen)
    live = ~frozen
    step = np.where(frozen, np.abs(delta) * (1.0 + 0.5 * np.abs(delta)), 0.0)
    step[live] = np.abs(np.linalg.inv(hessian[np.ix_(live, live)])) @ np.full(live.sum(), 2 * bar)
    assert np.all(np.abs(delta[live]) <= step[live])

    J, edf = _fit_sensitivities(grams)
    eta = [
        model._dm.matvec(model.result.beta) + model.result.intercept for model in (leverage, grams)
    ]
    assert np.all(np.abs(eta[0] - eta[1]) <= np.abs(J) @ step)
    assert abs(leverage.result.effective_df - grams.result.effective_df) <= np.abs(edf) @ step


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
