"""Regression tests for the remaining spline penalty coordinate maps (#659)."""

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, LambdaPolicy, Spline, SuperGLM, SuperLSS, s
from superglm.distributional import GaussianLS
from superglm.features.interaction import TensorInteraction


@pytest.mark.parametrize("kind", ["cr", "cr_cardinal", "ps", "bs", "ns"])
def test_tensor_margin_root_matches_independent_formed_penalty(kind):
    """On evenly spaced knots the ordinary congruence is a well-conditioned reference."""
    x = np.linspace(-2.0, 2.0, 120)
    spec = Spline(kind=kind, n_knots=5)
    spec.build(x)
    info = spec.tensor_marginal_ingredients(x)
    expected = info.projection.T @ spec._build_penalty() @ info.projection
    scale = np.linalg.norm(expected, 2)
    np.testing.assert_allclose(
        info.penalty_root.T @ info.penalty_root, expected, atol=1e-14 * scale
    )
    assert info.penalty_nullity == (0 if kind == "ns" else 1)
    assert not info.penalty_root[:, : info.penalty_nullity].any()


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("orders", [2, (1, 2), (2, 3)])
def test_selected_smooth_keeps_its_line_at_fixed_curvature(discrete, orders):
    """A huge wiggliness penalty leaves the least-squares line when selection is off."""
    x = np.geomspace(1.0, 1e5 if orders == (2, 3) else 1e7, 160)
    frame = pd.DataFrame({"x": x})
    y = 1 + x / x.max() + 0.05 * np.cos(np.log(x))
    policies = {"null": LambdaPolicy.off()}
    if orders == 2:
        policies["wiggle"] = LambdaPolicy.fixed(1e13)
    else:
        policies.update(
            {f"d{k}": LambdaPolicy.fixed(1e13) if k >= 2 else LambdaPolicy.off() for k in orders}
        )
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0,
        discrete=discrete,
        features={
            "x": Spline(
                kind="cr",
                n_knots=6,
                m=orders,
                select=True,
                knot_strategy="quantile_rows",
                lambda_policy=policies,
            )
        },
    ).fit_reml(frame, y)
    line = np.column_stack([np.ones_like(x), x / x.max()])
    np.testing.assert_allclose(
        model.predict(frame), line @ np.linalg.lstsq(line, y, rcond=None)[0], atol=2e-8, rtol=0
    )
    for key, omega in model._dm.group_matrices[0].omega_components:
        if key in ("wiggle", "d2", "d3"):
            assert not omega[0].any()


@pytest.mark.parametrize("decompose", [False, True])
@pytest.mark.parametrize("discrete", [False, True])
def test_tensor_keeps_the_bilinear_surface_under_both_margin_penalties(decompose, discrete):
    """A line times a line costs exactly zero on both centred margins."""
    x = np.geomspace(1.0, 1e7, 200)
    z = np.random.default_rng(17).permutation(np.linspace(0, 1, len(x)))
    specs = {"x": Spline(n_knots=6, knot_strategy="quantile_rows"), "z": Spline(n_knots=4)}
    for spec, values in zip(specs.values(), (x, z), strict=True):
        spec.build(values)
    tensor = TensorInteraction("x", "z", decompose=decompose)
    infos = (
        tensor.build_discrete(x, z, specs, (256, 256)).infos
        if discrete
        else tensor.build(x, z, specs)
    )
    wiggly = infos[1] if decompose else infos
    for margin in (tensor._marginal1, tensor._marginal2):
        assert np.linalg.norm(margin.penalty_root, 2) == pytest.approx(1, abs=5e-15)
    for (_, omega), repeat in zip(wiggly.penalty_components, (tensor._p2, tensor._p1), strict=True):
        assert np.sum(~np.any(omega, axis=0)) == repeat - int(decompose)
    coefficients = np.zeros(tensor._p1 * tensor._p2)
    coefficients[0] = 1
    if decompose:
        np.testing.assert_array_equal(infos[0].projection[:, 0], coefficients)
    curve = tensor._centered_marginal_dense(x, tensor._marginal1)[:, 0]
    other = tensor._centered_marginal_dense(z, tensor._marginal2)[:, 0]
    expected = (x - x.mean()) / x.max() * (z - z.mean())
    np.testing.assert_allclose(
        curve * other,
        expected * np.dot(expected, curve * other) / np.dot(expected, expected),
        atol=2e-13,
    )

    model = SuperGLM(family="gaussian", selection_penalty=0, discrete=discrete, features=specs)
    model._add_interaction("x", "z", decompose=decompose)
    y = 1 + x / x.max() + z + expected + np.random.default_rng(13).normal(0, 0.1, len(x))
    model.fit_reml(pd.DataFrame({"x": x, "z": z}), y)
    penalties = [item for item in model._reml_penalties if item.group_name.startswith("x:z")]
    assert len(penalties) == 2
    for item, repeat in zip(penalties, (tensor._p2, tensor._p1), strict=True):
        assert np.sum(~np.any(item.omega_ssp, axis=0)) == repeat - int(decompose)


@pytest.mark.parametrize("discrete", [False, True])
def test_by_level_curve_retains_its_unpenalized_slope(discrete):
    x = np.geomspace(1.0, 1e5, 200)
    labels = np.tile(["A", "B"], 100)
    frame = pd.DataFrame({"x": x, "g": labels})
    y = 1 + x / x.max() * (1 + (labels == "B")) + 0.01 * np.sin(np.log(x))
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0,
        discrete=discrete,
        features={"x": Spline(n_knots=6, knot_strategy="quantile_rows"), "g": Categorical()},
        interactions=[("x", "g")],
    ).fit_reml(frame, y)
    (level,) = [p for p in model._reml_penalties if p.group_name.startswith("x:g[")]
    group = model._dm.group_matrices[level.group_index]
    assert not group.omega[0].any()
    assert not level.omega_ssp[0].any()
    curve = group.toarray()[group.row_idx, 0]
    line = np.column_stack((np.ones(len(curve)), x[group.row_idx] / x.max()))
    np.testing.assert_allclose(
        curve,
        line @ np.linalg.lstsq(line, curve, rcond=None)[0],
        atol=2e-13 * np.max(np.abs(curve)),
    )


@pytest.mark.parametrize("degree, order, nullity", [(2, 2, 1), (3, 4, 1), (3, 2, 0)])
def test_natural_difference_penalty_counts_only_lines_left_by_its_boundaries(
    degree, order, nullity
):
    x = np.geomspace(1.0, 1e5, 160)
    frame = pd.DataFrame({"x": x})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0,
        features={
            "x": Spline(
                kind="ns",
                n_knots=8,
                m=order,
                degree=degree,
                knot_strategy="quantile_rows",
                lambda_policy=LambdaPolicy.fixed(1e14),
            )
        },
    ).fit_reml(frame, np.log(x) + 0.1 * np.cos(x))
    group = model._dm.group_matrices[0]
    assert group.R_inv_local is not None
    assert sum(~np.any(model._reml_penalties[0].omega_ssp, axis=0)) == nullity
    assert model.result.effective_df == pytest.approx(1 + nullity, abs=1e-7)


@pytest.mark.parametrize("selected", [False, True])
def test_distributional_location_keeps_its_unpenalized_line(selected):
    x = np.geomspace(1.0, 1e5, 180)
    frame = pd.DataFrame({"x": x})
    y = 1 + x / 1e5 + np.random.default_rng(42).normal(scale=0.1, size=len(x))
    family = GaussianLS()
    policy = {"wiggle": LambdaPolicy.fixed(1e14)}
    if selected:
        policy["null"] = LambdaPolicy.off()
    model = SuperLSS(
        family,
        family.location(
            s(
                "x",
                kind="cr",
                n_knots=8,
                select=selected,
                knot_strategy="quantile_rows",
                lambda_policy=policy,
            )
        ),
        family.scale(),
    ).fit(frame, y)
    penalty = next(p for p in model._require_fitted().layout.penalties if "wiggle" in p.name)
    assert not penalty.omega_ssp[0].any()
    line = np.column_stack([np.ones_like(x), x / x.max()])
    np.testing.assert_allclose(
        model.predict(frame), line @ np.linalg.lstsq(line, y, rcond=None)[0], atol=1e-7, rtol=0
    )
