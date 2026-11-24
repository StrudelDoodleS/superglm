"""Regression tests for the remaining spline penalty coordinate maps (#659)."""

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, LambdaPolicy, Numeric, Spline, SuperGLM, SuperLSS, s
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


@pytest.mark.parametrize(
    ("kind", "boundary", "x"),
    [
        ("ps", (0.0, 1.0), np.full(50, 0.5)),
        ("ps", (0.0, 1.0), np.linspace(0.0, 0.3, 50)),
        ("bs", None, np.full(50, 0.5)),
    ],
    ids=["one-value", "unreached-basis", "zero-penalty"],
)
def test_tensor_margin_its_rows_cannot_identify_keeps_the_formed_penalty(kind, boundary, x):
    """A margin states its penalty's null coordinates only where its rows identify them.

    On one value the centred basis is zero, and with a boundary past the rows a
    basis function has no row: an exactly unpenalised tensor coordinate would
    then be fitted to the rounding of its column (a coefficient of 4.3e14 where
    one kernel left 1e-18 there). Such a margin keeps the data's coordinates and
    the formed penalty, as a main effect does, and so does a ``bs`` margin on
    one value, whose penalty is zero.
    """
    spec = Spline(kind=kind, n_knots=3, boundary=boundary)
    spec.build(x)
    info = spec.tensor_marginal_ingredients(x)
    assert info.penalty_root is None and info.penalty_nullity is None
    formed = info.projection.T @ spec._build_penalty() @ info.projection
    np.testing.assert_array_equal(info.penalty, formed)


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("discrete", [False, True])
def test_precision_weights_decide_whether_a_tensor_margin_states_its_null_direction(discrete):
    """Pin a tensor fit against the same fit without its two rows of weight 1e-23.

    Only those two rows show the ``x`` margin's slope, and ``v`` is the
    tensor's bilinear direction. A margin asking the weights it is centred on
    would state that slope as an exact null coordinate beside ``v``, and the
    fitted values would move by 3e-2.
    """
    a = np.linspace(-1, 1, 80)
    rng = np.random.default_rng(2026)
    spread = rng.uniform(0, 1, 80)
    x = np.r_[0.5 + 1e-9 * a, 0.0, 1.0]
    z = np.r_[spread, 0.2, 0.8]
    signal = 1 + np.sin(3 * spread) + 0.3 * a * (spread - 0.5)
    y = np.r_[signal + 0.05 * rng.normal(size=80), 0.0, 2.0]
    weights = np.r_[np.ones(80), 1e-23, 1e-23]
    frame = pd.DataFrame({"x": x, "z": z, "v": (x - 0.5) * (z - z.mean()) / 1e-9})

    def fitted(rows):
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0,
            spline_penalty=1e-6,
            discrete=discrete,
            features={
                "x": Spline(kind="ps", n_knots=2, boundary=(0, 1)),
                "z": Spline(kind="ps", n_knots=2, boundary=(0, 1)),
                "v": Numeric(),
            },
            interactions=[("x", "z")],
        ).fit(frame.iloc[rows], y[rows], sample_weight=weights[rows])
        return model.predict(frame.iloc[:80])

    np.testing.assert_allclose(fitted(slice(None)), fitted(slice(80)), atol=1e-6, rtol=0)


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("discrete", [False, True])
def test_precision_weights_decide_whether_a_level_states_its_null_direction(discrete):
    """Pin a level's rank and weighted least-squares line with negligible-weight endpoints.

    Level B's rows sit within 1e-9 of one value but for two of weight 1e-23,
    and ``u`` is its line. A level asking its knot-geometry weights would
    state that line as an exact null coordinate beside ``u``: rank 7 for 6,
    and a curve 8e-3 off the line.
    """
    a = np.linspace(-1, 1, 80)
    spread = np.linspace(0, 1, 80)
    x = np.r_[spread, 0.5 + 1e-9 * a, 0.0, 1.0]
    labels = np.r_[np.full(80, "A"), np.full(82, "B")]
    y = np.r_[1 + spread, 1 + 0.4 * a + 0.1 * np.cos(np.arange(80)), 0.0, 2.0]
    weights = np.r_[np.ones(160), 1e-23, 1e-23]
    frame = pd.DataFrame({"x": x, "g": labels, "u": np.where(labels == "B", (x - 0.5) / 1e-9, 0.0)})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0,
        spline_penalty=1e-6,
        discrete=discrete,
        features={
            "x": Spline(kind="bs", n_knots=0, degree=2, boundary=(-1, 2)),
            "g": Categorical(),
            "u": Numeric(),
        },
        interactions=[("x", "g")],
    ).fit(frame, y, sample_weight=weights)
    design = np.column_stack([np.ones(80), a])
    line = design @ np.linalg.lstsq(design, y[80:160], rcond=None)[0]
    np.testing.assert_allclose(model.predict(frame)[80:160], line, rtol=1e-6)
    assert model.result.reml_hessian_rank == 6


@pytest.mark.pins_behaviour
def test_decomposed_tensor_fits_where_one_margin_penalty_outweighs_the_data():
    """Pin the surface one dominant margin penalty leaves, across the turn into its eigenbasis.

    With the other margin's penalty off, a weight of 1e12 outweighs the data
    by more than ``1 / sqrt(u)`` and the block is turned into the penalty's
    eigenbasis; at 1e6 it is not. The surface moves by ``0.2 / weight``, so
    the two fits agree to 1e-6. A tensor in its margins' fixed coordinates
    declares no null basis, and takes that turn as it did before them.
    """
    rng = np.random.default_rng(7)
    x, z = rng.uniform(0, 1, 400), rng.uniform(0, 1, 400)
    frame = pd.DataFrame({"x": x, "z": z})
    y = 1 + x * z + 0.3 * x * np.sin(4 * z) + 0.05 * rng.normal(size=400)

    def fit(weight):
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0,
            discrete=False,
            spline_penalty={
                "x": 1.0,
                "z": 1.0,
                "x:z:wiggly:margin_x": weight,
                "x:z:wiggly:margin_z": 0.0,
            },
            features={"x": Spline(kind="ps", n_knots=6), "z": Spline(kind="ps", n_knots=6)},
        )
        model._add_interaction("x", "z", decompose=True)
        model.fit(frame, y)
        block = next(
            matrix
            for matrix, group in zip(model._dm.group_matrices, model._groups, strict=True)
            if group.name == "x:z:wiggly"
        )
        turn = block.R_inv_local
        return model.predict(frame), not np.array_equal(turn, np.eye(len(turn)))

    stiff, stiff_turned = fit(1e12)
    loose, loose_turned = fit(1e6)
    assert stiff_turned and not loose_turned
    np.testing.assert_allclose(stiff, loose, atol=1e-6, rtol=0)


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
