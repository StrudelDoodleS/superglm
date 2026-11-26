"""Regression tests for the remaining spline penalty coordinate maps (#659)."""

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.linalg
import scipy.optimize

import superglm._lambda_box as lambda_box
from superglm import Categorical, LambdaPolicy, Numeric, Spline, SuperGLM, SuperLSS, s
from superglm._group_matrix import _group_matrix_range as group_ranges
from superglm.distributional import GaussianLS
from superglm.dm_builder import rebuild_design_matrix_with_lambdas
from superglm.features.interaction import TensorInteraction
from superglm.group_matrix import (
    DiscretizedSplineCategoricalGroupMatrix,
    SplineCategoricalGroupMatrix,
    SupportCompressedSplineCategoricalGroupMatrix,
)
from superglm.model.reml_setup import collect_reml_groups
from superglm.reml.convergence import newton_step_stop_bar
from superglm.reml.penalty_algebra import build_penalty_components


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


def _reciprocal_rows():
    """Rows with ``x z = 1`` on values symmetric about zero, and a response linear in both."""
    x = np.tile(np.r_[-np.linspace(1, 4, 12), np.linspace(1, 4, 12)], 3)
    z = 1 / x
    y = 2 + x / 4 + z + 0.2 * np.random.default_rng(3).normal(size=72)
    return x, z, y


def _stiff_tensor_fit(kind, discrete, x, z, y, weights=None):
    return SuperGLM(
        family="gaussian",
        selection_penalty=0,
        spline_penalty=1e13,
        discrete=discrete,
        features={"x": Spline(kind=kind, n_knots=3), "z": Spline(kind=kind, n_knots=3)},
        interactions=[("x", "z")],
    ).fit(pd.DataFrame({"x": x, "z": z}), y, sample_weight=weights)


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("weightless_rows", [False, True])
def test_tensor_null_product_the_intercept_takes_is_not_fitted(discrete, weightless_rows):
    """Pin the rank, EDF and least-squares line where the margins' lines multiply to a constant.

    Each ``cr`` margin's rows identify its line, but the tensor's bilinear
    direction is ``x z = 1``, the intercept. Stated as an exact null
    coordinate, its column is the rounding centring leaves, fitted as a 25th
    direction with a coefficient near 1e15 and fitted values up to 8e-2 off.
    Such a tensor keeps the data's coordinates and its formed penalty, where
    the fit's rank drops the direction as it did before those coordinates.
    Two rows off the curve at weight zero identify nothing more.
    """
    x, z, y = _reciprocal_rows()
    rows = pd.DataFrame({"x": x, "z": z})
    design = np.column_stack([np.ones_like(x), x, z])
    line = design @ np.linalg.lstsq(design, y, rcond=None)[0]
    weights = None
    if weightless_rows:
        x, z, y = np.r_[x, -2.0, 3.0], np.r_[z, 0.75, 0.5], np.r_[y, 0.0, 5.0]
        weights = np.r_[np.ones(72), 0.0, 0.0]
    model = _stiff_tensor_fit("cr", discrete, x, z, y, weights)
    np.testing.assert_allclose(model.predict(rows), line, atol=1e-9, rtol=0)
    assert model.result.reml_hessian_rank == 24
    assert model.result.effective_df == pytest.approx(3, abs=1e-7)


@pytest.mark.parametrize("discrete", [False, True])
def test_tensor_null_product_its_rows_nearly_cancel_stays_an_exact_coordinate(discrete):
    """A null product within 1.4e-4 of a constant is still identified, and fitted.

    A ``ps`` margin's null curve is the spline of a linear coefficient
    sequence, not a line on quantile knots, so on the same rows the product
    of the two curves varies by 1.4e-4 of its size: four digits lost, not the
    eight that would leave rounding. It stays an exact null coordinate, and a
    stiff penalty leaves least squares on the intercept, both curves and
    their product, a fourth direction the formed penalty loses.
    """
    x, z, y = _reciprocal_rows()
    model = _stiff_tensor_fit("ps", discrete, x, z, y)
    curves = []
    for name, values in (("x", x), ("z", z)):
        raw = np.asarray(model._specs[name]._raw_basis_matrix(values))
        curves.append(raw @ np.arange(raw.shape[1], dtype=float))
    design = np.column_stack([np.ones_like(x), *curves, curves[0] * curves[1]])
    surface = design @ np.linalg.lstsq(design, y, rcond=None)[0]
    np.testing.assert_allclose(
        model.predict(pd.DataFrame({"x": x, "z": z})), surface, atol=1e-8, rtol=0
    )
    assert model.result.reml_hessian_rank == 49
    assert model.result.effective_df == pytest.approx(4, abs=1e-7)


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


@pytest.mark.parametrize("dominant", ["x", "z"])
@pytest.mark.parametrize("kind", ["ps", "cr"])
def test_turned_fixed_coordinate_tensor_stores_its_penalty_in_the_turned_coordinates(
    kind, dominant
):
    """REML's penalty for a turned tensor is ``C' Omega C``, with its rank and log-determinant.

    A dominant margin turns the block into that margin's eigenbasis ``C``,
    a signed permutation of the fixed coordinates, so each margin's penalty
    in the solver's coordinates is still diagonal and is formed exactly.
    """
    rng = np.random.default_rng(7)
    x, z = rng.uniform(0, 1, 400), rng.uniform(0, 1, 400)
    frame = pd.DataFrame({"x": x, "z": z})
    y = 1 + x * z + 0.3 * x * np.sin(4 * z) + 0.05 * rng.normal(size=400)
    other = "z" if dominant == "x" else "x"
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0,
        discrete=False,
        spline_penalty={
            "x": 1.0,
            "z": 1.0,
            f"x:z:wiggly:margin_{dominant}": 1e12,
            f"x:z:wiggly:margin_{other}": 0.0,
        },
        features={"x": Spline(kind=kind, n_knots=6), "z": Spline(kind=kind, n_knots=6)},
    )
    model._add_interaction("x", "z", decompose=True)
    model.fit(frame, y)
    matrices = list(model._dm.group_matrices)
    index = next(i for i, group in enumerate(model._groups) if group.name == "x:z:wiggly")
    turn = matrices[index].R_inv_local
    assert not np.array_equal(turn, np.eye(len(turn)))
    groups = collect_reml_groups(list(model._groups), matrices)
    penalties = [
        penalty
        for penalty in build_penalty_components(matrices, groups)
        if penalty.group_index == index
    ]
    assert len(penalties) == 2
    for penalty in penalties:
        formed = turn.T @ penalty.omega_raw @ turn
        np.testing.assert_array_equal(penalty.omega_ssp, formed)
        spectrum = np.linalg.eigvalsh(penalty.omega_raw)
        spectrum = spectrum[spectrum > 1e-12 * spectrum[-1]]
        assert penalty.rank == spectrum.size
        np.testing.assert_allclose(penalty.log_det_omega_plus, np.sum(np.log(spectrum)), rtol=1e-12)


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


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize(("kind", "orders"), [("bs", (2, 1)), ("bs", (3, 2)), ("cr", (2, 1))])
def test_by_level_curve_of_a_multi_order_parent_keeps_its_penalty_null_space(
    kind, orders, discrete
):
    """A level's one penalty has its parent's first order, and its null polynomials stay unpenalised.

    Under stiff penalties each level's fit is its least-squares polynomial of
    degree below ``orders[0]``. Counting the null space every listed order
    shares declared the level's penalty one rank too high, so the level stated
    no null coordinates and the weight of 1e13 bent its polynomial.
    """
    x1 = np.geomspace(1.0, 1e5, 200)
    x = np.r_[x1, x1]
    labels = np.r_[np.full(200, "A"), np.full(200, "B")]
    t = x / x.max()
    level_b = labels == "B"
    y = 1 + t * (1 + level_b) + 0.05 * np.sin(np.log(x)) + 0.03 * level_b * np.cos(3 * np.log(x))
    frame = pd.DataFrame({"x": x, "g": labels})
    first = orders[0]
    penalty = {f"x:d{order}": 1e13 if order == first else 0.0 for order in orders}
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0,
        discrete=discrete,
        spline_penalty={**penalty, "x:g[B]": 1e13},
        features={
            "x": Spline(kind=kind, n_knots=6, m=orders, knot_strategy="quantile_rows"),
            "g": Categorical(),
        },
        interactions=[("x", "g")],
    ).fit(frame, y)

    (group,) = [
        matrix
        for matrix, info in zip(model._dm.group_matrices, model._groups, strict=True)
        if info.name == "x:g[B]"
    ]
    predicted = model.predict(frame)
    for rows in (~level_b, level_b):
        design = np.vander(t[rows], first, increasing=True)
        fitted = design @ np.linalg.lstsq(design, y[rows], rcond=None)[0]
        np.testing.assert_allclose(predicted[rows], fitted, atol=1e-10, rtol=0)
    assert not group.omega[: first - 1].any()


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


@pytest.mark.parametrize("discrete", [False, True])
def test_distributional_centering_keeps_a_selected_smooths_penalty_roots(monkeypatch, discrete):
    """Centring a selected smooth's rows leaves REML the penalty it read, roots included.

    The centred block keeps the source's penalty and coordinate map, so it
    carries the roots beside them, as the SuperGLM path's row subsets, ranges
    and rebuilds do. No built-in selected smooth states roots yet, so the
    source is given the exact roots of its own diagonal components. Without
    them the centred block sends REML to the eigen route instead.
    """
    from superglm.distributional import predictor

    calls = []
    center = predictor._center_selected_smooths

    def record(compiled, geometry_weight, *, intercept):
        calls.append((compiled, geometry_weight))
        return center(compiled, geometry_weight, intercept=intercept)

    monkeypatch.setattr(predictor, "_center_selected_smooths", record)
    x = np.geomspace(1.0, 1e5, 180)
    y = 1 + x / 1e5 + np.random.default_rng(42).normal(scale=0.1, size=len(x))
    family = GaussianLS()
    policy = {"wiggle": LambdaPolicy.fixed(1.0), "null": LambdaPolicy.fixed(1.0)}
    SuperLSS(
        family,
        family.location(
            s("x", kind="cr", n_knots=8, select=True, lambda_policy=policy),
        ),
        family.scale(),
        discrete=discrete,
    ).fit(pd.DataFrame({"x": x}), y)

    compiled, geometry_weight = calls[0]
    (index,) = [i for i, group in enumerate(compiled.groups) if group.feature_name == "x"]
    source = compiled.design.group_matrices[index]
    roots = {}
    for suffix, omega in source.omega_components:
        diagonal = np.diag(omega)
        assert np.count_nonzero(omega) == np.count_nonzero(diagonal)
        rows = np.flatnonzero(diagonal)
        roots[suffix] = np.zeros((rows.size, len(diagonal)))
        roots[suffix][np.arange(rows.size), rows] = np.sqrt(diagonal[rows])
    source.term_roots = roots
    centered = center(compiled, geometry_weight, intercept=True)
    assert centered.design.group_matrices[index].term_roots is roots

    def penalties(design):
        matrices = list(design.group_matrices)
        groups = collect_reml_groups(list(compiled.groups), matrices)
        return build_penalty_components(matrices, groups, certified=True)

    expected, actual = penalties(compiled.design), penalties(centered.design)
    assert [p.name for p in actual] == [p.name for p in expected]
    for got, want in zip(actual, expected, strict=True):
        assert (got.rank, got.log_det_omega_plus) == (want.rank, want.log_det_omega_plus)
        np.testing.assert_array_equal(got.eigvals_omega, want.eigvals_omega)
        np.testing.assert_array_equal(got.omega_ssp, want.omega_ssp)


def _green_silverman_cardinal_root(knots: np.ndarray) -> np.ndarray:
    """Independent Green & Silverman (1994, ch. 2) natural cubic spline penalty root E (E' E = S)."""
    k = np.asarray(knots, dtype=np.float64)
    K = len(k)
    hbar = (k[-1] - k[0]) / (K - 1)
    u = (k - k[0]) / hbar
    h = np.diff(u)
    Q = np.zeros((K, K - 2), dtype=np.float64)
    R = np.zeros((K - 2, K - 2), dtype=np.float64)
    for j in range(K - 2):
        Q[j, j] = 1.0 / h[j]
        Q[j + 1, j] = -1.0 / h[j] - 1.0 / h[j + 1]
        Q[j + 2, j] = 1.0 / h[j + 1]
        R[j, j] = (h[j] + h[j + 1]) / 3.0
        if j < K - 3:
            R[j, j + 1] = R[j + 1, j] = h[j + 1] / 6.0
    L = scipy.linalg.cholesky(R, lower=True)
    return scipy.linalg.solve_triangular(L, Q.T, lower=True)


@pytest.mark.parametrize(
    ("repeats", "discrete", "expected_cls"),
    [
        (1, False, SplineCategoricalGroupMatrix),
        (20, False, SupportCompressedSplineCategoricalGroupMatrix),
        (1, True, DiscretizedSplineCategoricalGroupMatrix),
    ],
    ids=["exact-csr", "exact-compressed", "binned"],
)
def test_by_level_reml_penalty_spectrum_matches_main_effect(repeats, discrete, expected_cls):
    """A by-level curve carries its penalty root to REML and matches the main effect's spectrum."""
    u = 2.0**-53
    x1 = np.geomspace(1.0, 1e5, 200)
    xr = np.tile(x1, repeats)
    x = np.r_[xr, xr]
    labels = np.r_[np.full(xr.size, "A"), np.full(xr.size, "B")]
    t = np.log(x) / np.log(x.max())
    rng = np.random.default_rng(5)
    y = np.where(
        labels == "A",
        1.0 + t + np.sin(4.0 * t),
        2.0 + 3.0 * t + 2.0 * np.sin(4.0 * t),
    ) + 0.01 * rng.normal(size=x.size)
    frame = pd.DataFrame({"x": x, "g": labels})

    model = SuperGLM(
        family="gaussian",
        selection_penalty=0,
        discrete=discrete,
        features={
            "x": Spline(kind="cr_cardinal", n_knots=20, knot_strategy="quantile_rows"),
            "g": Categorical(),
        },
        interactions=[("x", "g")],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(
            frame,
            y,
            lambda2_init={"x": 1e-6, "x:g[B]": 1e-6},
            reml_tol=1e-10,
            pirls_tol=1e-10,
        )

    assert model._reml_result.converged
    by_name = {pc.name: pc for pc in model._reml_penalties}
    main_pc, level_pc = by_name["x"], by_name["x:g[B]"]
    main_gm = model._dm.group_matrices[main_pc.group_index]
    level_gm = model._dm.group_matrices[level_pc.group_index]
    assert type(level_gm) is expected_cls
    assert model._reml_result.lambdas["x"] == pytest.approx(
        model._reml_result.lambdas["x:g[B]"], rel=1e-12
    )
    assert main_pc.rank == 20
    assert level_pc.rank == 20

    ev_main = np.sort(main_pc.eigvals_omega)
    ev_level = np.sort(level_pc.eigvals_omega)
    # Perturbation of singular values squared under Cholesky whitening R_inv_local:
    # cond(R_inv_local)^2 scales the Cholesky factor rounding in G + lam*Omega + ridge.
    cond_whitening = float(np.linalg.cond(main_gm.R_inv_local))
    rtol = 128.0 * u * (cond_whitening**2)
    np.testing.assert_allclose(ev_level, ev_main, rtol=rtol, atol=0.0)
    assert level_pc.log_det_omega_plus == pytest.approx(
        main_pc.log_det_omega_plus, abs=len(ev_main) * rtol
    )


@pytest.mark.parametrize("discrete", [False, True], ids=["exact", "binned"])
def test_by_level_binding_penalty_matches_dense_root_reml_reference(discrete):
    """Where a level's penalty binds on graded knots, REML matches an independent root oracle."""
    x1 = np.geomspace(1.0, 1e5, 200)
    x = np.r_[x1, x1]
    labels = np.r_[np.full(200, "A"), np.full(200, "B")]
    s_coord = x / x.max()
    rng = np.random.default_rng(19)
    y = np.where(
        labels == "A",
        1.0 + 0.8 * s_coord + 0.5 * np.sin(2.5 * np.pi * s_coord),
        1.5
        + 1.4 * s_coord
        + 0.5 * np.sin(2.5 * np.pi * s_coord)
        + 0.6 * np.cos(2.0 * np.pi * s_coord),
    ) + 0.15 * rng.normal(size=400)
    frame = pd.DataFrame({"x": x, "g": labels})

    model = SuperGLM(
        family="gaussian",
        selection_penalty=0,
        discrete=discrete,
        features={
            "x": Spline(kind="cr_cardinal", n_knots=20, knot_strategy="quantile_rows"),
            "g": Categorical(),
        },
        interactions=[("x", "g")],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y, reml_tol=1e-10, pirls_tol=1e-10)

    result = model._reml_result
    assert result.converged
    by_name = {pc.name: pc for pc in model._reml_penalties}
    assert by_name["x"].rank == 20
    assert by_name["x:g[B]"].rank == 20

    lam_x = float(result.lambdas["x"])
    lam_b = float(result.lambdas["x:g[B]"])
    assert 1e-2 < lam_x < 1e4
    assert 1e-2 < lam_b < 1e4

    # Independent Wood (2011) Gaussian REML reference using Green & Silverman's
    # rectangular penalty root transported into the fitted design's coordinates.
    E_raw = _green_silverman_cardinal_root(model._specs["x"]._cr_knots)
    gm_x, gm_g, gm_b = model._dm.group_matrices
    X_full = np.column_stack([np.ones(len(y)), gm_x.toarray(), gm_g.toarray(), gm_b.toarray()])
    E_x = E_raw @ gm_x.R_inv
    E_b = E_raw @ gm_b.R_inv
    sv_x = scipy.linalg.svdvals(E_x)
    sv_b = scipy.linalg.svdvals(E_b)
    sv_x = sv_x[sv_x > 0.0]
    sv_b = sv_b[sv_b > 0.0]
    logdet_x = 2.0 * float(np.sum(np.log(sv_x)))
    logdet_b = 2.0 * float(np.sum(np.log(sv_b)))
    n = len(y)
    M_p = X_full.shape[1] - (len(sv_x) + len(sv_b))
    p_x = gm_x.shape[1]

    def dense_root_reml(rho: np.ndarray) -> tuple[float, np.ndarray]:
        lx, lb = float(np.exp(rho[0])), float(np.exp(rho[1]))
        Ax = np.zeros((len(E_x), X_full.shape[1]), dtype=np.float64)
        Ax[:, 1 : 1 + p_x] = np.sqrt(lx) * E_x
        Ab = np.zeros((len(E_b), X_full.shape[1]), dtype=np.float64)
        Ab[:, 2 + p_x :] = np.sqrt(lb) * E_b
        A = np.vstack([X_full, Ax, Ab])
        rhs = np.concatenate([y, np.zeros(len(E_x) + len(E_b))])
        Q, Rq = np.linalg.qr(A, mode="reduced")
        coef = scipy.linalg.solve_triangular(Rq, Q.T @ rhs)
        resid = A @ coef - rhs
        Dp = float(resid @ resid)
        logdet_H = 2.0 * float(np.sum(np.log(np.abs(np.diag(Rq)))))
        logdet_S = len(sv_x) * np.log(lx) + logdet_x + len(sv_b) * np.log(lb) + logdet_b
        phi = Dp / (n - M_p)
        V = 0.5 * ((n - M_p) * np.log(2.0 * np.pi * phi) + (n - M_p) + logdet_H - logdet_S)
        return V, coef

    opt = scipy.optimize.minimize(
        lambda rho: dense_root_reml(rho)[0],
        np.array([np.log(lam_x), np.log(lam_b)]),
        method="Nelder-Mead",
        options={"xatol": 1e-12, "fatol": 1e-13, "maxiter": 2000},
    )
    V_ref, coef_ref = dense_root_reml(opt.x)
    lam_x_ref, lam_b_ref = float(np.exp(opt.x[0])), float(np.exp(opt.x[1]))
    pred_ref = X_full @ coef_ref
    pred_fit = model.predict(frame)

    assert lam_x == pytest.approx(lam_x_ref, rel=1e-5)
    assert lam_b == pytest.approx(lam_b_ref, rel=1e-5)
    assert result.objective == pytest.approx(V_ref, abs=1e-8)
    np.testing.assert_allclose(pred_fit[200:], pred_ref[200:], atol=1e-6, rtol=0.0)


def _by_level_dense_reml(model, y):
    """Wood's (2011) Gaussian REML in the two log lambdas of a fitted main-plus-level model.

    Independent of the fit's own REML: one QR of the design over Green and
    Silverman's rectangular penalty roots, in the fitted design's coordinates.
    """
    root = _green_silverman_cardinal_root(model._specs["x"]._cr_knots)
    main, factor, level = model._dm.group_matrices
    design = np.column_stack([np.ones(len(y)), main.toarray(), factor.toarray(), level.toarray()])
    width = design.shape[1]
    roots = [(1, root @ main.R_inv), (width - level.shape[1], root @ level.R_inv)]
    spectra = [scipy.linalg.svdvals(block) for _, block in roots]
    dof = len(y) - width + sum(map(len, spectra))
    rhs = np.r_[y, np.zeros(sum(len(block) for _, block in roots))]

    def criterion(*rho):
        rows = [np.zeros((len(block), width)) for _, block in roots]
        for row, (start, block), log_lambda in zip(rows, roots, rho, strict=True):
            row[:, start : start + block.shape[1]] = np.exp(0.5 * log_lambda) * block
        q, r = np.linalg.qr(np.vstack([design, *rows]))
        residual = rhs - q @ (q.T @ rhs)
        log_det = 2.0 * np.sum(np.log(np.abs(np.diag(r)))) - sum(
            len(values) * log_lambda + 2.0 * np.sum(np.log(values))
            for values, log_lambda in zip(spectra, rho, strict=True)
        )
        return 0.5 * (dof * (np.log(2.0 * np.pi * (residual @ residual) / dof) + 1.0) + log_det)

    return criterion


@pytest.mark.parametrize("discrete", [False, True], ids=["exact", "binned"])
@pytest.mark.parametrize("side", ["floor", "ceiling"])
def test_reml_finds_a_levels_optimum_beside_a_main_effect_its_bound_holds(
    side, discrete, monkeypatch
):
    """A smoothing parameter held at a bound leaves the other's Newton step alone (#735).

    The main effect sits on a bound of the box with its REML score pointing out
    of it, and its Hessian coupling to the level's term is negative; an
    independent dense criterion confirms both. A Newton step solved with the
    held term and clipped to the box afterwards moved the level's term uphill,
    and the search ended unconverged, short of the optimum that criterion places.
    """
    x = np.tile(np.geomspace(1.0, 1e5, 200), 2)
    level_b = np.repeat([False, True], 200)
    if side == "floor":
        t = np.log(x) / np.log(x.max())
        mean = 1.0 + t + np.sin(4.0 * t) + level_b * (1.0 + 2.0 * t + 0.5 * np.cos(3.0 * t))
        y = mean + 0.1 * np.random.default_rng(5).normal(size=400)
    else:
        # The main effect is a line, whose optimum lies beyond a ceiling of 3.
        monkeypatch.setattr(lambda_box, "LAMBDA_CEILING", 3.0)
        t = x / x.max()
        mean = 1.0 + 0.8 * t + level_b * (0.5 + 0.6 * t + 0.6 * np.cos(2.0 * np.pi * t))
        y = mean + 0.15 * np.random.default_rng(19).normal(size=400)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0,
        discrete=discrete,
        features={
            "x": Spline(kind="cr_cardinal", n_knots=20, knot_strategy="quantile_rows"),
            "g": Categorical(),
        },
        interactions=[("x", "g")],
    )
    tolerance = 1e-10
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(
            pd.DataFrame({"x": x, "g": np.where(level_b, "B", "A")}),
            y,
            reml_tol=tolerance,
            pirls_tol=tolerance,
        )

    result = model._reml_result
    assert result.converged
    assert result.bound_pressure == (("x", side),)
    box = result.lambda_box
    assert box.side(result.lambdas["x"]) == side
    held, free = np.log(result.lambdas["x"]), np.log(result.lambdas["x:g[B]"])

    criterion = _by_level_dense_reml(model, y)
    h = 1e-2
    assert criterion(held + (h if side == "floor" else -h), free) > criterion(held, free)
    coupling = (
        criterion(held + h, free + h)
        - criterion(held + h, free - h)
        - criterion(held - h, free + h)
        + criterion(held - h, free - h)
    )
    assert coupling < 0.0
    reference = scipy.optimize.minimize_scalar(
        lambda rho: criterion(held, rho),
        bounds=(box.log_floor, box.log_ceiling),
        method="bounded",
        options={"xatol": 1e-10},
    )
    assert free == pytest.approx(reference.x, abs=newton_step_stop_bar(tolerance))
    assert result.objective == pytest.approx(
        reference.fun, abs=tolerance * (1 + abs(reference.fun))
    )


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("discrete", [False, True], ids=["exact", "binned"])
def test_reml_started_beside_the_floor_steps_onto_it(discrete):
    """Pin a start inside the stop's window of the floor ending on the floor (#735).

    The stop reads an outward score within 0.01 log units of a bound as zero.
    A search that takes no step there either certifies its own start: held out
    of the Newton system with no step of its own, a term started 0.009 log
    units above the floor was returned there, 0.22 nats above the criterion a
    start on the floor and one 0.011 above it both reach. The clipped Newton
    step this replaces already moved it onto the floor.
    """
    x = np.tile(np.geomspace(1.0, 1e5, 200), 2)
    level_b = np.repeat([False, True], 200)
    t = np.log(x) / np.log(x.max())
    mean = 1.0 + t + np.sin(4.0 * t) + level_b * (1.0 + 2.0 * t + 0.5 * np.cos(3.0 * t))
    y = mean + 0.1 * np.random.default_rng(5).normal(size=400)
    frame = pd.DataFrame({"x": x, "g": np.where(level_b, "B", "A")})
    tolerance = 1e-10

    def stop(start):
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0,
            discrete=discrete,
            features={
                "x": Spline(kind="cr_cardinal", n_knots=20, knot_strategy="quantile_rows"),
                "g": Categorical(),
            },
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit_reml(
                frame,
                y,
                lambda2_init={"x": start},
                reml_tol=tolerance,
                pirls_tol=tolerance,
            )
        return model._reml_result

    on_floor = stop(lambda_box.LAMBDA_FLOOR)
    box = on_floor.lambda_box
    assert on_floor.converged
    assert on_floor.bound_pressure == (("x", "floor"),)
    for offset in (0.009, 0.011):
        result = stop(box.floor * np.exp(offset))
        assert result.converged
        assert box.side(result.lambdas["x"]) == "floor"
        assert result.objective == pytest.approx(
            on_floor.objective, abs=tolerance * (1 + abs(on_floor.objective))
        )


@pytest.mark.parametrize(
    ("repeats", "discrete", "expected_cls"),
    [
        (1, False, SplineCategoricalGroupMatrix),
        (20, False, SupportCompressedSplineCategoricalGroupMatrix),
        (1, True, DiscretizedSplineCategoricalGroupMatrix),
    ],
    ids=["exact-csr", "exact-compressed", "binned"],
)
def test_by_level_term_roots_survive_subset_range_pickle_and_extreme_whitening(
    repeats, discrete, expected_cls, monkeypatch
):
    """By-level term_roots survive subsetting, ranges, pickle/legacy-unpickle, and QR whitening."""
    import superglm.dm_builder as dm_builder

    x1 = np.geomspace(1.0, 1e5, 200)
    xr = np.tile(x1, repeats)
    x = np.r_[xr, xr]
    labels = np.r_[np.full(xr.size, "A"), np.full(xr.size, "B")]
    s_coord = x / x.max()
    y = np.where(labels == "A", 1.0 + s_coord, 2.0 + 3.0 * s_coord) + 0.05 * np.sin(
        4.0 * np.pi * s_coord
    )
    frame = pd.DataFrame({"x": x, "g": labels})

    # Force _whitening_map's LinAlgError fallback on the formed-Gram Cholesky
    # attempt (deterministic across BLAS builds) while letting both the QR
    # fallback's cholesky(G + ridge) and the eigh fallback's cholesky(rotated)
    # run, so dropping _whitening_penalty drops the 6 weak modes via eigh.
    orig_whitening_map = dm_builder._whitening_map
    orig_cholesky = np.linalg.cholesky
    fail_next_large = [False]

    def forced_cholesky(a, *args, **kwargs):
        if fail_next_large[0] and float(np.max(np.abs(a))) > 1e12:
            fail_next_large[0] = False
            raise np.linalg.LinAlgError("forced indefinite formed penalty")
        return orig_cholesky(a, *args, **kwargs)

    def wrapped_whitening_map(G, penalty, lam2):
        fail_next_large[0] = True
        try:
            return orig_whitening_map(G, penalty, lam2)
        finally:
            fail_next_large[0] = False

    monkeypatch.setattr(np.linalg, "cholesky", forced_cholesky)
    monkeypatch.setattr(dm_builder, "_whitening_map", wrapped_whitening_map)

    model = SuperGLM(
        family="gaussian",
        selection_penalty=0,
        spline_penalty=1e14,
        discrete=discrete,
        features={
            "x": Spline(kind="cr_cardinal", n_knots=20, knot_strategy="quantile_rows"),
            "g": Categorical(),
        },
        interactions=[("x", "g")],
    ).fit(frame, y)

    gm_level = model._dm.group_matrices[2]
    group_level = model._groups[2]
    assert type(gm_level) is expected_cls
    (pc,) = build_penalty_components(model._dm.group_matrices, [(2, group_level)])
    assert pc.rank == 20
    # QR whitening via term_roots keeps the null row exact zero and bounds all 20
    # whitened eigenvalues by 1 / lam (unlike eigh(omega), which drops the 6 weak modes).
    assert not pc.omega_ssp[0].any()
    assert float(np.max(pc.eigvals_omega)) <= 1.01 / 1e14
    assert gm_level.term_roots is not None

    # Rebuild at another extreme lambda keeps term_roots and rank 20 via QR whitening.
    rebuilt_dm = rebuild_design_matrix_with_lambdas(
        model._dm,
        model._groups,
        {"x": 1e14, "x:g[B]": 2e14},
        np.ones(len(y), dtype=np.float64),
    )
    gm_rebuilt = rebuilt_dm.group_matrices[2]
    assert type(gm_rebuilt) is expected_cls
    (pc_rebuilt,) = build_penalty_components(rebuilt_dm.group_matrices, [(2, group_level)])
    assert pc_rebuilt.rank == 20
    assert not pc_rebuilt.omega_ssp[0].any()
    assert float(np.max(pc_rebuilt.eigvals_omega)) <= 1.01 / 2e14
    assert gm_rebuilt.term_roots is not None

    # row_subset, group_row_range, and pickle preserve term_roots.
    sub = gm_level.row_subset(np.arange(0, len(y), 2))
    assert sub.term_roots is gm_level.term_roots
    ranged = group_ranges.group_row_range(gm_level, 0, len(y) // 2)
    assert ranged is not None
    assert ranged.term_roots is gm_level.term_roots
    restored = pickle.loads(pickle.dumps(gm_level))
    assert restored.term_roots is not None
    np.testing.assert_array_equal(restored.term_roots[""], gm_level.term_roots[""])

    # Unpickling a legacy state without term_roots restores term_roots as None.
    dict_state, slot_state = gm_level.__getstate__()
    slot_state.pop("term_roots")
    legacy = expected_cls.__new__(expected_cls)
    legacy.__setstate__((dict_state, slot_state))
    assert legacy.term_roots is None


def test_outer_clipped_separated_level_rechecks_null_identification():
    """An outer-clipped separated level re-runs _rows_identify on its clipped rows."""
    rng = np.random.default_rng(7)
    x_grid = np.linspace(0.0, 1.0, 100)
    ya = rng.poisson(2.0, size=100).astype(float)
    yb = np.zeros(100, dtype=float)
    yb[50] = 3.0  # One positive response: outer clipping collapses level B onto x = x_grid[50].
    frame = pd.DataFrame(
        {"x": np.r_[x_grid, x_grid], "g": np.r_[np.full(100, "A"), np.full(100, "B")]}
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SuperGLM(
            family="poisson",
            selection_penalty=0,
            discrete=False,
            features={"x": Spline(kind="ps", n_knots=6), "g": Categorical()},
            interactions=[("x", "g")],
        ).fit(frame, np.r_[ya, yb])

    gm_level = model._dm.group_matrices[2]
    assert gm_level.R_inv_local is None
    assert gm_level.term_roots is None
