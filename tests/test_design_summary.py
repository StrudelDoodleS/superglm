"""Read-only fitted-design route summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, Constraint, Numeric, PSpline, Spline, SuperGLM
from superglm.group_matrix import (
    DiscretizedSCOPGroupMatrix,
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
    DiscretizedTensorGroupMatrix,
    SparseSSPGroupMatrix,
    SplineCategoricalGroupMatrix,
)

EXPECTED_COLUMNS = [
    "term",
    "feature",
    "solver_start",
    "solver_end",
    "n_columns",
    "representation",
    "compressed",
    "storage_rows",
    "ordinary_tabmat_partition",
    "specialised_discrete_route",
    "route_reason",
]


def _design_data(n: int = 180):
    rng = np.random.default_rng(20260720)
    x = rng.uniform(0.0, 1.0, n)
    z = rng.uniform(-1.0, 1.0, n)
    category = np.resize(np.array(["A", "B", "C"], dtype=object), n)
    y = 0.4 + 0.7 * x - 0.25 * z + 0.15 * (category == "B")
    y += rng.normal(0.0, 0.04, n)
    return pd.DataFrame({"x": x, "z": z, "category": category}), y


def _fit_representation(case: str):
    X, y = _design_data()
    common = {"family": "gaussian", "selection_penalty": 0.0}
    if case == "sparse-ssp":
        model = SuperGLM(
            **common,
            features={"x": Spline(n_knots=6, penalty="ssp")},
        ).fit(X[["x"]], y)
        return model, SparseSSPGroupMatrix, "sparse-ssp", False, None
    if case == "spline-categorical":
        model = SuperGLM(
            **common,
            features={
                "x": Spline(n_knots=6, penalty="ssp"),
                "category": Categorical(base="first"),
            },
            interactions=[("x", "category")],
        ).fit(X[["x", "category"]], y)
        return (
            model,
            SplineCategoricalGroupMatrix,
            "spline-categorical",
            False,
            None,
        )
    if case == "discretized-ssp":
        model = SuperGLM(
            **common,
            discrete=True,
            n_bins=14,
            features={"x": Spline(n_knots=6, penalty="ssp")},
        ).fit(X[["x"]], y)
        return model, DiscretizedSSPGroupMatrix, "discretized-ssp", True, "binned-ssp"
    if case == "discretized-scop":
        model = SuperGLM(
            **common,
            discrete=True,
            n_bins=14,
            features={
                "x": PSpline(
                    n_knots=6,
                    constraint=Constraint.fit.increasing,
                )
            },
        ).fit(X[["x"]], y)
        return model, DiscretizedSCOPGroupMatrix, "discretized-scop", True, "binned-scop"
    if case == "discretized-spline-categorical":
        model = SuperGLM(
            **common,
            discrete=True,
            n_bins=14,
            features={
                "x": Spline(n_knots=6, penalty="ssp"),
                "category": Categorical(base="first"),
            },
            interactions=[("x", "category")],
        ).fit(X[["x", "category"]], y)
        return (
            model,
            DiscretizedSplineCategoricalGroupMatrix,
            "discretized-spline-categorical",
            True,
            "binned-spline-categorical",
        )
    if case == "discretized-tensor":
        model = SuperGLM(
            **common,
            discrete=True,
            n_bins={"x": 14, "z": 12},
            features={
                "x": Spline(n_knots=6, penalty="ssp"),
                "z": Spline(n_knots=6, penalty="ssp"),
            },
            interactions=[("x", "z")],
        ).fit(X[["x", "z"]], y)
        return (
            model,
            DiscretizedTensorGroupMatrix,
            "discretized-tensor",
            True,
            "observed-tensor-support",
        )
    raise AssertionError(f"unknown case: {case}")


def _expected_storage_rows(group) -> int:
    if type(group) is SparseSSPGroupMatrix:
        return int(group.B.shape[0])
    if type(group) is SplineCategoricalGroupMatrix:
        return int(group.B_level.shape[0])
    if type(group) is DiscretizedSCOPGroupMatrix:
        return int(group.B_scop_unique.shape[0])
    return int(group.B_unique.shape[0])


def test_design_summary_rejects_unfitted_and_released_designs():
    with pytest.raises(
        RuntimeError,
        match=r"Model must be fitted before calling design_summary\(\)\.",
    ):
        SuperGLM(features={"x": Numeric()}).design_summary()

    X, y = _design_data()
    compact = SuperGLM(
        family="gaussian",
        features={"x": Numeric()},
        selection_penalty=0.0,
        retain_fit_state=False,
    ).fit(X[["x"]], y)
    assert compact._dm is None

    with pytest.raises(RuntimeError, match="retain_fit_state=False.*discarded the fitted design"):
        compact.design_summary()


def test_design_summary_numeric_and_categorical_is_lazy_and_read_only(monkeypatch):
    X, y = _design_data()
    model = SuperGLM(
        family="gaussian",
        features={"x": Numeric(), "category": Categorical(base="first")},
        selection_penalty=0.0,
    ).fit(X[["x", "category"]], y)

    from superglm.group_matrix import DesignMatrix

    model._dm = DesignMatrix(
        list(model._dm.group_matrices),
        n=model._dm.n,
        p=model._dm.p,
    )
    before = (
        model._dm._tabmat_built,
        model._dm._mixed_bin_space_centering_plan_attempted,
        model._dm.raw_spline_tabmat_plan_built,
    )
    assert before == (False, False, False)
    for matrix_type in {type(group) for group in model._dm.group_matrices}:
        monkeypatch.setattr(
            matrix_type,
            "toarray",
            lambda _self: pytest.fail("design_summary must not materialize matrix rows"),
        )

    first = model.design_summary()
    after = (
        model._dm._tabmat_built,
        model._dm._mixed_bin_space_centering_plan_attempted,
        model._dm.raw_spline_tabmat_plan_built,
    )
    second = model.design_summary()

    assert first is not second
    pd.testing.assert_frame_equal(first, second)
    assert first.columns.tolist() == EXPECTED_COLUMNS
    assert first["term"].tolist() == ["x", "category"]
    assert first["feature"].tolist() == ["x", "category"]
    assert first["representation"].tolist() == ["dense", "categorical-codes"]
    assert first["compressed"].tolist() == [False, False]
    assert first["storage_rows"].tolist() == [len(X), len(X)]
    assert first["ordinary_tabmat_partition"].tolist() == [False, False]
    assert first["specialised_discrete_route"].tolist() == [None, None]
    assert first["route_reason"].tolist() == ["categorical-layout", "categorical-layout"]
    assert after == before


@pytest.mark.parametrize(
    "case",
    [
        "sparse-ssp",
        "spline-categorical",
        "discretized-ssp",
        "discretized-scop",
        "discretized-spline-categorical",
        "discretized-tensor",
    ],
)
def test_design_summary_reports_specialised_storage_without_materializing(case, monkeypatch):
    model, matrix_type, representation, compressed, discrete_route = _fit_representation(case)
    matching = [
        (index, group)
        for index, group in enumerate(model._dm.group_matrices)
        if type(group) is matrix_type
    ]
    assert matching
    before = (
        model._dm._tabmat_built,
        model._dm._mixed_bin_space_centering_plan_attempted,
        model._dm.raw_spline_tabmat_plan_built,
    )
    monkeypatch.setattr(
        matrix_type,
        "toarray",
        lambda _self: pytest.fail("design_summary must not materialize matrix rows"),
    )

    summary = model.design_summary()

    after = (
        model._dm._tabmat_built,
        model._dm._mixed_bin_space_centering_plan_attempted,
        model._dm.raw_spline_tabmat_plan_built,
    )
    assert summary.columns.tolist() == EXPECTED_COLUMNS
    assert after == before
    for index, group in matching:
        row = summary.iloc[index]
        assert row["solver_start"] == model._groups[index].start
        assert row["solver_end"] == model._groups[index].end
        assert row["n_columns"] == model._groups[index].size
        assert row["representation"] == representation
        assert bool(row["compressed"]) is compressed
        assert row["storage_rows"] == _expected_storage_rows(group)
        assert not bool(row["ordinary_tabmat_partition"])
        assert row["specialised_discrete_route"] == discrete_route
        expected_reason = (
            "contains-compressed-group" if compressed else "specialised-group-representation"
        )
        assert row["route_reason"] == expected_reason
