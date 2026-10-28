import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

import superglm.solvers.scop_exact_support as scop_exact_support
from superglm import Constraint, PSpline, SuperGLM
from superglm.solvers.scop_exact_support import build_exact_scop_support


def test_build_exact_scop_support_reproduces_row_level_products_exactly():
    B = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    support = build_exact_scop_support(B)
    assert support is not None

    W = np.array([1.0, 2.0, 1.5, 0.5, 3.0])
    z = np.array([0.2, 0.4, 0.1, 0.3, 0.6])

    BtWB_row = B.T @ (B * W[:, None])
    BtWz_row = B.T @ (W * z)

    assert_allclose(support.B_unique[support.row_to_support], B, rtol=0.0, atol=0.0)

    # Aggregate the weights over the support the way the solver does.
    W_agg = np.bincount(support.row_to_support, weights=W, minlength=support.n_unique)
    Wz_agg = np.bincount(support.row_to_support, weights=W * z, minlength=support.n_unique)
    BtWB_support = support.B_unique.T @ (support.B_unique * W_agg[:, None])
    BtWz_support = support.B_unique.T @ Wz_agg

    assert_allclose(BtWB_support, BtWB_row)
    assert_allclose(BtWz_support, BtWz_row)


def test_build_exact_scop_support_is_numpys_row_unique():
    """The keyed grouping returns np.unique's rows, order and inverse, collisions included."""
    rng = np.random.default_rng(3)
    values = rng.integers(0, 40, size=5000).astype(float) / 7.0
    smooth = np.column_stack([np.tanh(values - k) for k in range(6)])
    # rows [2, 0] and [0, 1] share the key 2 * 1 + 0 * 2 = 0 * 1 + 1 * 2
    colliding = np.array([[2.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    for B in (smooth, colliding):
        support = build_exact_scop_support(B)
        expected_rows, expected_inverse = np.unique(B, axis=0, return_inverse=True)
        np.testing.assert_array_equal(support.B_unique, expected_rows)
        np.testing.assert_array_equal(support.row_to_support, expected_inverse)


def test_build_exact_scop_support_returns_none_when_no_repeated_rows():
    B = np.eye(5)
    support = build_exact_scop_support(B)
    assert support is None


def test_single_scop_feature_support_compression_matches_fallback(monkeypatch):
    x = np.repeat(np.linspace(0.0, 1.0, 40), 5)
    y = 0.4 + 0.7 * x + 1.2 * x**2
    df = pd.DataFrame({"x": x})

    fast = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": PSpline(n_knots=10, constraint=Constraint.fit.convex, shape_engine="scop")},
    ).fit(df, y)

    monkeypatch.setattr(scop_exact_support, "build_exact_scop_support", lambda B: None)

    slow = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": PSpline(n_knots=10, constraint=Constraint.fit.convex, shape_engine="scop")},
    ).fit(df, y)

    assert_allclose(fast.result.beta, slow.result.beta, atol=1e-8, rtol=1e-8)
    assert_allclose(fast.predict(df), slow.predict(df), atol=1e-8, rtol=1e-8)


def test_single_scop_feature_support_compression_preserves_predictions(monkeypatch):
    x = np.repeat(np.linspace(0.0, 1.0, 50), 8)
    y = 0.4 + 0.7 * x + 1.2 * x**2
    df = pd.DataFrame({"x": x})

    fast = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": PSpline(n_knots=10, constraint=Constraint.fit.convex, shape_engine="scop")},
    ).fit(df, y)

    monkeypatch.setattr(
        scop_exact_support,
        "build_exact_scop_support",
        lambda B_scop: None,
    )

    slow = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": PSpline(n_knots=10, constraint=Constraint.fit.convex, shape_engine="scop")},
    ).fit(df, y)

    assert_allclose(fast.predict(df), slow.predict(df), atol=1e-8, rtol=1e-8)


def test_single_scop_feature_reuses_support_compression_when_available():
    x = np.repeat(np.linspace(0.0, 1.0, 30), 10)
    y = 0.4 + 0.7 * x + 1.2 * x**2
    df = pd.DataFrame({"x": x})

    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": PSpline(n_knots=10, constraint=Constraint.fit.convex, shape_engine="scop")},
    ).fit(df, y)

    scop_states = getattr(model._result, "scop_states", None)
    assert scop_states is not None
    state = next(iter(scop_states.values()))
    assert state["bin_idx"] is not None
    assert state["B_scop"].shape[0] < len(df)


def test_multi_scop_feature_fit_does_not_expose_exact_support_state():
    x1 = np.repeat(np.linspace(0.0, 1.0, 20), 6)
    x2 = np.repeat(np.linspace(1.0, 2.0, 20), 6)
    y = 0.3 + 0.4 * x1**2 + 0.2 * x2**2
    df = pd.DataFrame({"x1": x1, "x2": x2})

    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "x1": PSpline(n_knots=10, constraint=Constraint.fit.convex, shape_engine="scop"),
            "x2": PSpline(n_knots=10, constraint=Constraint.fit.convex, shape_engine="scop"),
        },
    ).fit(df, y)

    assert getattr(model._result, "scop_states", None) is None
