from __future__ import annotations

import inspect
import pickle
from typing import Any

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm._frame import as_eager_frame
from superglm._predictor_compiler import compile_predictor_design
from superglm.dm_builder import build_design_matrix
from superglm.features import Categorical, Numeric, Spline


def _compiled_snapshot(model: SuperGLM) -> dict[str, Any]:
    plan = model._dm.execution_plan
    return {
        "groups": tuple(
            (group.name, group.size, group.penalized, group.feature_name, group.subgroup_type)
            for group in model._groups
        ),
        "matrix_types": tuple(type(matrix).__name__ for matrix in model._dm.group_matrices),
        "penalty_components": tuple(
            None
            if getattr(matrix, "omega_components", None) is None
            else tuple(suffix for suffix, _omega in matrix.omega_components)
            for matrix in model._dm.group_matrices
        ),
        "ordinary_partition": (
            plan._ordinary_partition_reason,
            tuple(sorted(plan._ordinary_indices)),
        ),
        "link": type(model._link).__name__,
    }


def test_scalar_dense_compilation_snapshot_preserves_weighted_offset_prediction() -> None:
    n = 18
    x = np.linspace(-1.0, 1.0, n)
    z = np.linspace(0.05, 0.95, n)
    category = np.array(["a", "b", "c"] * 6)
    frame = pd.DataFrame({"x": x, "cat": category, "z": z})
    weights = np.linspace(0.8, 1.7, n)
    offset = np.linspace(-0.15, 0.12, n)
    response = (
        1.25
        + 0.7 * x
        + 0.25 * (category == "b")
        - 0.1 * (category == "c")
        + 0.2 * np.sin(3.0 * z)
        + offset
    )
    model = SuperGLM(
        family="gaussian",
        link="identity",
        features={
            "x": Numeric(),
            "cat": Categorical(base="first"),
            "z": Spline(n_knots=4, degree=2, penalty="ssp", select=True),
        },
        interactions=[("x", "cat")],
        selection_penalty=0.0,
        spline_penalty=0.2,
    )

    model.fit(frame, response, sample_weight=weights, offset=offset)

    assert _compiled_snapshot(model) == {
        "groups": (
            ("x", 1, True, "x", None),
            ("cat", 2, True, "cat", None),
            ("z", 6, True, "z", None),
            ("x:cat", 2, True, "x:cat", None),
        ),
        "matrix_types": (
            "DenseGroupMatrix",
            "CategoricalGroupMatrix",
            "SparseSSPGroupMatrix",
            "DenseGroupMatrix",
        ),
        "penalty_components": (None, None, ("null", "wiggle"), None),
        "ordinary_partition": ("contains-compressed-group", ()),
        "link": "IdentityLink",
    }
    np.testing.assert_allclose(
        model.predict(frame, offset=offset),
        np.array(
            [
                0.4342773971169359,
                0.8123397671747348,
                0.5893470797426734,
                0.8098200884437351,
                1.1832716947227435,
                0.95300020601887,
                1.1657917869414158,
                1.526831442187724,
                1.282995911778832,
                1.48338243260908,
                1.828388351666495,
                1.5682773558599465,
                1.755509292709795,
                2.0856823203580124,
                1.8113963547922558,
                1.9893561496252077,
                2.3108712886426224,
                2.030623054087032,
            ]
        ),
        rtol=0.0,
        atol=5e-12,
    )


def test_scalar_discrete_compilation_snapshot_preserves_prediction() -> None:
    n = 18
    z = np.linspace(0.0, 1.0, n)
    frame = pd.DataFrame({"z": z})
    weights = np.linspace(0.8, 1.7, n)
    offset = np.linspace(-0.15, 0.12, n)
    response = 0.8 + 0.5 * np.sin(4.0 * z) + offset
    model = SuperGLM(
        family="gaussian",
        link="identity",
        features={"z": Spline(n_knots=5, degree=2, penalty="ssp")},
        discrete=True,
        n_bins=7,
        selection_penalty=0.0,
        spline_penalty=0.2,
    )

    model.fit(frame, response, sample_weight=weights, offset=offset)

    assert _compiled_snapshot(model) == {
        "groups": (("z", 7, True, "z", None),),
        "matrix_types": ("DiscretizedSSPGroupMatrix",),
        "penalty_components": (None,),
        "ordinary_partition": ("contains-compressed-group", ()),
        "link": "IdentityLink",
    }
    np.testing.assert_allclose(
        model.predict(frame, offset=offset),
        np.array(
            [
                0.660324424462687,
                0.777795217518543,
                0.891778062725478,
                1.002022228045582,
                1.096994039734917,
                1.170425196845689,
                1.221861664748707,
                1.245400993264478,
                1.239227043876233,
                1.20459927422611,
                1.149074430166936,
                1.07391196934085,
                0.982815019383858,
                0.887818745112986,
                0.789848928437236,
                0.690052671663101,
                0.590540643034533,
                0.49135872664379,
            ]
        ),
        rtol=0.0,
        atol=5e-12,
    )


def test_independent_builds_do_not_mutate_caller_or_prior_compiled_state() -> None:
    """Two compilations must not reach each other, or the caller.

    The joint path compiles one predictor per parameter from specs it does not
    own, and a REML rebuild recompiles the same specs repeatedly, so learned
    state leaking between builds would silently couple predictors that the
    model treats as independent.
    """
    caller_spec = Spline(n_knots=5, degree=2, penalty="ssp")
    specs = {"x": caller_spec}
    caller_before = pickle.dumps(caller_spec)

    def compile_one(values: np.ndarray):
        n = len(values)
        return compile_predictor_design(
            as_eager_frame(pd.DataFrame({"x": values})),
            np.linspace(0.2, 1.1, n),
            geometry_weight=np.linspace(0.2, 1.1, n),
            polynomial_weight=np.linspace(0.2, 1.1, n),
            categorical_reporting_weight=np.linspace(0.2, 1.1, n),
            ordered_reporting_weight=np.linspace(0.2, 1.1, n),
            specs=specs,
            feature_order=["x"],
            interaction_specs={},
            interaction_order=[],
            pending_interactions=[],
            model_discrete=False,
            n_bins_config=32,
            lambda2=0.1,
        )

    first = compile_one(np.linspace(0.0, 1.0, 16))
    first_state = pickle.dumps(first.specs["x"])
    compile_one(np.linspace(-2.0, 3.0, 23))

    assert (
        pickle.dumps(caller_spec) == caller_before,
        pickle.dumps(first.specs["x"]) == first_state,
    ) == (True, True)

    assert isinstance(first.groups, tuple)
    with pytest.raises(TypeError):
        first.specs["other"] = Numeric()  # type: ignore[index]


def test_scalar_build_keeps_its_documented_in_place_spec_contract() -> None:
    """The two entry points deliberately disagree about who owns learned state.

    ``compile_predictor_design`` clones because a frozen predictor may not write
    to specs it was handed.  ``build_design_matrix`` documents in-place mutation
    and master's callers read learned state back off their own spec objects, so
    it keeps that contract and skips a deepcopy per REML rebuild and CV fold.
    Asserting both here keeps the difference deliberate rather than latent.
    """
    caller_spec = Spline(n_knots=5, degree=2, penalty="ssp")
    values = np.linspace(0.0, 1.0, 16)
    n = len(values)
    result = build_design_matrix(
        as_eager_frame(pd.DataFrame({"x": values})),
        np.linspace(0.2, 1.1, n),
        np.ones(n),
        None,
        family="gaussian",
        link_spec="identity",
        specs={"x": caller_spec},
        feature_order=["x"],
        interaction_specs={},
        interaction_order=[],
        pending_interactions=[],
        model_discrete=False,
        n_bins_config=32,
        lambda2=0.1,
        weight_semantics="frequency",
    )

    assert result.compiled.specs["x"] is caller_spec


def test_compiler_signature_excludes_scalar_likelihood_inputs() -> None:
    parameters = inspect.signature(compile_predictor_design).parameters

    assert {"y", "family", "link", "link_spec", "offset"}.isdisjoint(parameters)
