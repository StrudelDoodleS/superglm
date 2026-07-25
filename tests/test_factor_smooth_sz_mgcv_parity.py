"""Pinned clean-room parity against mgcv 1.9-4 ``bs="sz"``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from superglm import FactorSmooth, Spline, SuperGLM
from superglm.factor_smooth_geometry import sum_to_zero_penalty
from superglm.group_matrix import FactorSmoothGroupMatrix

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "factor_smooth_sz_mgcv_reference.json"
_CASE_NAMES = ("gaussian", "poisson", "poisson_discrete")


@pytest.fixture(scope="module")
def mgcv_sz_fixture() -> dict:
    return json.loads(_FIXTURE_PATH.read_text())


def _fortran_matrix(flat: list[float], dimensions: list[int]) -> np.ndarray:
    return np.asarray(flat, dtype=np.float64).reshape(tuple(dimensions), order="F")


def test_sz_reference_fixture_is_versioned_and_has_shared_geometry(
    mgcv_sz_fixture: dict,
) -> None:
    metadata = mgcv_sz_fixture["metadata"]
    construction = mgcv_sz_fixture["construction"]

    assert metadata["r_version"] == "R version 4.5.3 (2026-03-11)"
    assert metadata["mgcv_version"] == "1.9.4"
    assert metadata["seed"] == 20260725
    assert metadata["sz_term"] == ('s(x, f, bs="sz", k=6, xt=list(bs="ps"), m=2, id=1)')
    assert construction["design_dim"] == [48, 18]
    assert construction["penalty_count"] == 1
    assert construction["penalty_rank"] == 12
    assert construction["nullity"] == 6
    assert construction["no_id_smoothing_parameter_count"] == 4
    assert construction["prediction_dim"] == [1, 18]


def test_sz_construction_matches_mgcv_sorted_contrast_coordinates(
    mgcv_sz_fixture: dict,
) -> None:
    construction = mgcv_sz_fixture["construction"]
    x = np.asarray(construction["data"]["x"], dtype=np.float64)
    group = np.asarray(construction["data"]["f"], dtype=object)
    spec = FactorSmooth(
        "x",
        group="f",
        basis="sz",
        kind="ps",
        k=6,
        m=2,
    )
    info = spec.build(x, group, {})
    design = spec.transform(x, group)
    reference_design = _fortran_matrix(
        construction["design_flat"],
        construction["design_dim"],
    )

    assert list(spec._levels) == construction["levels"]
    assert info.n_cols == 18
    np.testing.assert_allclose(design, reference_design, rtol=0.0, atol=2e-10)

    assert [name for name, _omega in info.repeated_penalty_components] == ["wiggle"]
    local_penalty = np.asarray(info.repeated_penalty_components[0][1], dtype=np.float64)
    public_penalty = sum_to_zero_penalty(local_penalty, len(spec._levels))
    reference_penalty = _fortran_matrix(
        construction["penalty_flat"],
        construction["penalty_dim"],
    )
    np.testing.assert_allclose(
        public_penalty / construction["penalty_scale"],
        reference_penalty,
        rtol=0.0,
        atol=2e-10,
    )
    assert np.linalg.matrix_rank(public_penalty) == construction["penalty_rank"]
    assert (
        public_penalty.shape[0] - np.linalg.matrix_rank(public_penalty) == construction["nullity"]
    )

    prediction_data = construction["prediction_data"]
    prediction_design = spec.transform(
        np.asarray([prediction_data["x"]], dtype=np.float64),
        np.asarray([prediction_data["f"]], dtype=object),
    )
    reference_prediction = _fortran_matrix(
        construction["prediction_flat"],
        construction["prediction_dim"],
    )
    np.testing.assert_allclose(
        prediction_design,
        reference_prediction,
        rtol=0.0,
        atol=2e-10,
    )

    beta = np.linspace(-0.4, 0.7, info.n_cols)
    grid = np.linspace(-0.9, 0.9, 17)
    level_effects = np.stack(
        [
            spec.score(
                grid,
                np.repeat(level, len(grid)),
                beta,
            )
            for level in spec._levels
        ]
    )
    np.testing.assert_allclose(level_effects.sum(axis=0), 0.0, atol=2e-13)


@pytest.fixture(scope="module")
def fitted_sz_cases(mgcv_sz_fixture: dict) -> dict[str, SuperGLM]:
    fitted: dict[str, SuperGLM] = {}
    for name in _CASE_NAMES:
        case = mgcv_sz_fixture[name]
        data = case["data"]
        X = pd.DataFrame({"x": data["x"], "f": data["f"]})
        y = np.asarray(data["y"], dtype=np.float64)
        offset = (
            None
            if "exposure" not in data
            else np.log(np.asarray(data["exposure"], dtype=np.float64))
        )
        model = SuperGLM(
            family="gaussian" if name == "gaussian" else "poisson",
            features={"x": Spline(kind="ps", k=7, m=2)},
            interactions=[
                FactorSmooth(
                    "x",
                    group="f",
                    basis="sz",
                    kind="ps",
                    k=6,
                    m=2,
                )
            ],
            selection_penalty=0.0,
            direct_solve="structured",
            discrete=name.endswith("_discrete"),
            n_bins=512,
            tol=1e-10,
            max_iter=200,
        )
        model.fit_reml(
            X,
            y,
            offset=offset,
            max_reml_iter=50,
            reml_tol=1e-9,
            pirls_tol=1e-10,
            max_pirls_iter=200,
            runtime_validation="skip",
        )
        fitted[name] = model
    return fitted


def _prediction_frame(case: dict) -> tuple[pd.DataFrame, np.ndarray | None]:
    values = case["prediction_data"]
    frame = pd.DataFrame({"x": values["x"], "f": values["f"]})
    offset = (
        None
        if "exposure" not in values
        else np.log(np.asarray(values["exposure"], dtype=np.float64))
    )
    return frame, offset


def _unseen_frame(case: dict) -> tuple[pd.DataFrame, np.ndarray | None]:
    values = case["unseen_data"]
    frame = pd.DataFrame(
        {
            "x": values["x"],
            "f": [f"unseen-{index}" for index in range(len(values["x"]))],
        }
    )
    offset = (
        None
        if "exposure" not in values
        else np.log(np.asarray(values["exposure"], dtype=np.float64))
    )
    return frame, offset


@pytest.mark.parametrize(
    ("name", "prediction_rtol", "deviance_rtol", "deviance_atol", "edf_atol"),
    [
        # The deterministic Gaussian residual scale is only 8.9e-4. A
        # 0.17% difference in the independently optimized global lambda moves
        # deviance by 4.7e-6 while predictions remain within 2.5e-5.
        ("gaussian", 1.0e-3, 5.0e-6, 5.0e-6, 5.0e-3),
        ("poisson", 1.0e-2, 5.0e-4, 3.0e-7, 1.0e-1),
        ("poisson_discrete", 1.2e-2, 8.0e-4, 3.0e-7, 1.2e-1),
    ],
)
def test_sz_fit_matches_mgcv_reml_and_freml(
    mgcv_sz_fixture: dict,
    fitted_sz_cases: dict[str, SuperGLM],
    name: str,
    prediction_rtol: float,
    deviance_rtol: float,
    deviance_atol: float,
    edf_atol: float,
) -> None:
    case = mgcv_sz_fixture[name]
    reference = case["reference"]
    model = fitted_sz_cases[name]
    prediction_frame, offset = _prediction_frame(case)
    report = model.factor_smooth("x:f:sz", grid=np.asarray(case["curve_grid"]))

    assert report.basis == "sz"
    assert report.lambdas["wiggle"] == pytest.approx(
        reference["unscaled_lambdas"]["sz_wiggle"],
        rel=5.0e-2,
    )
    assert model._reml_lambdas["x"] == pytest.approx(
        reference["unscaled_lambdas"]["global"],
        rel=5.0e-2,
    )
    assert model.result.deviance == pytest.approx(
        reference["deviance"],
        rel=deviance_rtol,
        abs=deviance_atol,
    )
    assert model.result.effective_df == pytest.approx(reference["total_edf"], abs=edf_atol)
    assert model._group_edf["x"] == pytest.approx(reference["global_edf"], abs=edf_atol)
    assert report.effective_df == pytest.approx(reference["sz_edf"], abs=edf_atol)
    assert model.result.phi == pytest.approx(reference["scale"], rel=8e-4, abs=3e-8)

    np.testing.assert_allclose(
        model.predict(prediction_frame, offset=offset),
        reference["conditional_prediction"],
        rtol=prediction_rtol,
        atol=4e-4,
    )
    np.testing.assert_allclose(
        model.predict(
            prediction_frame,
            offset=offset,
            random_effects="population",
        ),
        reference["global_only_prediction"],
        rtol=prediction_rtol,
        atol=4e-4,
    )
    actual_deviation = model._predict_eta_exact(
        prediction_frame,
        random_effects="conditional",
    ) - model._predict_eta_exact(
        prediction_frame,
        random_effects="population",
    )
    np.testing.assert_allclose(
        actual_deviation,
        reference["sz_deviation_link"],
        rtol=prediction_rtol,
        atol=3e-3,
    )
    np.testing.assert_allclose(
        report.curves["effect"],
        reference["sz_deviation_link"],
        rtol=prediction_rtol,
        atol=3e-3,
    )
    pivot = report.curves.pivot(index="x", columns="level", values="effect")
    np.testing.assert_allclose(pivot.sum(axis=1), 0.0, atol=2e-12)


@pytest.mark.parametrize("name", _CASE_NAMES)
def test_sz_unseen_population_matches_mgcv_term_exclusion(
    mgcv_sz_fixture: dict,
    fitted_sz_cases: dict[str, SuperGLM],
    name: str,
) -> None:
    case = mgcv_sz_fixture[name]
    model = fitted_sz_cases[name]
    frame, offset = _unseen_frame(case)
    conditional = model.predict(frame, offset=offset, random_effects="conditional")
    population = model.predict(frame, offset=offset, random_effects="population")

    np.testing.assert_allclose(conditional, population, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        population,
        case["reference"]["unseen_population_prediction"],
        rtol=1.2e-2,
        atol=4e-4,
    )


def test_sz_superglm_exact_and_discrete_predictions_match(
    mgcv_sz_fixture: dict,
    fitted_sz_cases: dict[str, SuperGLM],
) -> None:
    frame, offset = _prediction_frame(mgcv_sz_fixture["poisson"])
    exact = fitted_sz_cases["poisson"]
    discrete = fitted_sz_cases["poisson_discrete"]

    # mgcv's own exact-versus-discrete fixture differs by up to 0.351%
    # here, so this bound captures measured binning/fREML geometry rather than
    # demanding identity between the two algorithms.
    np.testing.assert_allclose(
        discrete.predict(frame, offset=offset),
        exact.predict(frame, offset=offset),
        rtol=3.6e-3,
        atol=4e-4,
    )
    assert discrete.result.deviance == pytest.approx(exact.result.deviance, rel=8e-4)


def test_discrete_sz_terminal_lambdas_are_the_evaluated_candidate(
    mgcv_sz_fixture: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = mgcv_sz_fixture["poisson_discrete"]
    data = case["data"]
    X = pd.DataFrame({"x": data["x"], "f": data["f"]})
    y = np.asarray(data["y"], dtype=np.float64)
    offset = np.log(np.asarray(data["exposure"], dtype=np.float64))
    original = FactorSmoothGroupMatrix.factor_smooth_discrete_cell_moments

    def equivalent_batched_moments(self, W, rhs):
        cell_weights, _gram, _xtw, _xt_rhs = original(self, W, rhs)
        cell_rhs = np.zeros_like(cell_weights)
        np.add.at(
            cell_rhs,
            (self.codes, self.bin_idx),
            np.asarray(rhs, dtype=np.float64),
        )
        effective_basis = np.ascontiguousarray(
            self.B_unique @ self.natural_map,
            dtype=np.float64,
        )
        weighted_basis = cell_weights[:, :, None] * effective_basis[None, :, :]
        local_gram = effective_basis.T[None, :, :] @ weighted_basis
        local_gram = 0.5 * (local_gram + local_gram.transpose(0, 2, 1))
        return (
            np.ascontiguousarray(cell_weights),
            np.ascontiguousarray(local_gram),
            np.ascontiguousarray(cell_weights @ effective_basis),
            np.ascontiguousarray(cell_rhs @ effective_basis),
        )

    monkeypatch.setattr(
        FactorSmoothGroupMatrix,
        "factor_smooth_discrete_cell_moments",
        equivalent_batched_moments,
    )
    model = SuperGLM(
        family="poisson",
        features={"x": Spline(kind="ps", k=7, m=2)},
        interactions=[
            FactorSmooth(
                "x",
                group="f",
                basis="sz",
                kind="ps",
                k=6,
                m=2,
            )
        ],
        selection_penalty=0.0,
        direct_solve="structured",
        discrete=True,
        n_bins=512,
        tol=1e-10,
        max_iter=200,
    )
    model.fit_reml(
        X,
        y,
        offset=offset,
        max_reml_iter=50,
        reml_tol=1e-9,
        pirls_tol=1e-10,
        max_pirls_iter=200,
        runtime_validation="skip",
    )

    result = model._reml_result
    assert result.converged
    assert result.lambda_history[-1] == result.lambda_history[-2]
    assert model._reml_lambdas["x:f:sz:wiggle"] == pytest.approx(
        case["reference"]["unscaled_lambdas"]["sz_wiggle"],
        rel=5.0e-2,
    )
