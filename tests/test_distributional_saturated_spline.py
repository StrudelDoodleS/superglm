"""Smoothing an isolated spline beside a certified coefficient alias (issue 576)."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import sympy

from superglm import GaussianLS, SuperLSS, s, term
from superglm.distributional.results.iteration import DistributionalEFSConfig
from superglm.distributional.smoothing import evidence
from superglm.distributional.smoothing.objective import _stable_isolated_gfs_update
from superglm.features import Numeric
from superglm.reml.efs_update import EFSComponentState, wood_fasiolo_update
from superglm.solvers.rank import decompose_symmetric_deflated
from superglm.types import LambdaPolicy


@pytest.fixture(scope="module")
def spline_book():
    rng = np.random.default_rng(4)
    u = np.repeat(rng.uniform(-1, 1, 256), 4)
    x = np.repeat(rng.normal(size=256), 4)
    z = x + 1e-7 * np.tile([1, 1, -1, -1], 256)
    y = 2 * u + np.sin(3 * u) + np.tile([1, -1, -1, 1], 256)
    frame = pd.DataFrame({"u": u, "x": x, "z": z})
    grid = pd.DataFrame({"u": np.linspace(u.min(), u.max(), 101), "x": 0.0, "z": 0.0})
    # The profiled minimum of V is at 2.6423. Fit the explicitly reduced
    # model there, independently of the smoothing search and alias selection.
    family = GaussianLS(scale_floor=0)
    reference = SuperLSS(
        family,
        family.location(s("u", kind="cr", k=8), term("x", Numeric())),
        family.scale(),
        discrete=False,
    ).fit(frame, y, lambdas={"location:u#wiggle": 2.6423})
    return frame, y, grid, reference.predict(grid)


@pytest.mark.parametrize("outer", ["efs", "efs+newton"])
@pytest.mark.parametrize("start,cancel", [(1.0, False), (1e10, False), (1e10, True)])
def test_an_unrelated_alias_preserves_the_interior_spline(
    spline_book, outer, start, cancel, monkeypatch
):
    frame, y, grid, reference_curve = spline_book
    if cancel:
        original = evidence.wood_fasiolo_update

        def rounded(components, *args, **kwargs):
            update = original(components, *args, **kwargs)
            (component,) = components
            if component.lambda_value != start:
                return update
            # Force both roundings that can occur in coefficient coordinates:
            # rank - λ tr(H⁻¹ S) = 0 and β'Sβ = 0. Every BLAS then presents
            # the same spurious fixed point to the actual smoothing search.
            name = component.name
            return replace(
                update,
                lambdas={name: start},
                quadratic_forms={name: 0.0},
                trace_terms={name: component.rank / start},
                raw_log_steps={name: 0.0},
                log_steps={name: 0.0},
                stationarity_log_residuals={name: 0.0},
                proposal_kinds={name: "fixed_point_fallback"},
            )

        monkeypatch.setattr(evidence, "wood_fasiolo_update", rounded)
    family = GaussianLS(scale_floor=0)
    model = SuperLSS(
        family,
        family.location(s("u", kind="cr", k=8), term("x", Numeric()), term("z", Numeric())),
        family.scale(),
        discrete=False,
    ).fit_reml(frame, y, initial_lambda=start, max_lambda=1e12, practical_reml=False, outer=outer)
    fitted = model._require_fitted().fit_state.smoothing
    assert fitted.converged
    assert fitted.lambdas["location:u#wiggle"] == pytest.approx(2.6423, abs=0.003)
    assert fitted.objective == pytest.approx(1476.94271, abs=1e-5)
    assert model.result_.total_effective_df == pytest.approx(8.531, abs=0.001)
    np.testing.assert_allclose(model.predict(grid), reference_curve, rtol=0, atol=5e-5)


def _certified_rank(formed, exact):
    exact = sympy.Matrix(exact).applyfunc(sympy.Rational)

    def project(basis):
        columns = sympy.Matrix(basis).applyfunc(sympy.Rational)
        cross = exact * columns
        block = np.asarray(columns.T * cross, dtype=float)
        cross = np.asarray(cross, dtype=float)
        return block, cross, 2.0**-52 * np.abs(block), 2.0**-52 * np.abs(cross)

    return decompose_symmetric_deflated(formed, project, formation_bound=1e-13)


def _saturated_alias(*, duplicate=True):
    # K keeps both penalized coordinates and one of two identical nuisance
    # columns. In the penalty range H = 1+λ and the exact numerator = 1/(1+λ).
    penalty = np.array([[0.5, -0.5], [-0.5, 0.5]])
    lam = 2.0**30
    data = np.eye(4)
    if duplicate:
        data[2:, 2:] = 1.0
    else:
        data[-1, -1] = 0.0
    h = data.copy()
    h[:2, :2] += lam * penalty

    rank = _certified_rank(h, h)
    assert rank.rank == 3 and rank.grey_subspace["certified"]
    beta = np.array([1 + 2.0**-24, 1 - 2.0**-24, 0, 0])
    component = EFSComponentState("s", slice(0, 2), penalty, 1.0, lam, LambdaPolicy.estimate())
    fit = SimpleNamespace(
        terminal_rank=rank,
        coefficients=beta,
        coefficient_face=None,
        terminal_curvature=SimpleNamespace(certified=True),
        terminal_data_curvature=data,
        terminal_penalized_curvature=h,
    )
    return component, fit


@pytest.mark.parametrize("duplicate", [True, False])
def test_the_reduced_spline_update_matches_its_scalar_eigenspace(duplicate):
    component, fit = _saturated_alias(duplicate=duplicate)
    inverse = fit.terminal_rank.pseudo_inverse()
    raw = wood_fasiolo_update((component,), fit.coefficients, inverse)
    update, names = _stable_isolated_gfs_update(
        (component,), fit, inverse, raw, DistributionalEFSConfig()
    )
    lam = component.lambda_value
    quadratic = 2.0**-47
    assert names == {"s"}
    # The saturated numerator is recovered to the absolute precision of the
    # rank-one trace. The log quotient amplifies that error by 1+λ.
    step = np.log(1.0 / ((1.0 + lam) * lam * quadratic))
    assert update.raw_log_steps["s"] == pytest.approx(step, abs=32 * np.finfo(float).eps * lam)
    assert update.quadratic_forms["s"] == pytest.approx(quadratic, rel=1e-8, abs=0)


def test_a_retained_row_direction_uses_its_own_curvature_beside_a_dropped_column():
    # Orthonormal h,x,t; the nuisance columns x and x+d(h+t) explain half
    # of h. Its conditional data information is 1/2. The last column is zero.
    lam, d = 2.0**28, 2.0**-22
    rows = np.array([[1, 0, d, 0], [0, 1, 1, 0], [0, 0, d, 0]])
    data = rows.T @ rows
    exact = data + np.diag([lam, 0, 0, 0])
    # Perturb the formed cross term within the supplied formation uncertainty.
    # The rows resolve it. Pairing the retained inverse with the formed data
    # counts the perturbation in the spline's update.
    data[0, 1] = data[1, 0] = 2.0**-47 * np.sqrt(lam)
    h = data + np.diag([lam, 0, 0, 0])
    rank = _certified_rank(h, exact)
    assert rank.rank == 3 and rank.grey_subspace["certified"] and np.any(rank.grey_columns)
    beta = np.array([1 / lam, 0, 0, 0])
    component = EFSComponentState("s", slice(0, 1), np.eye(1), 1, lam, LambdaPolicy.estimate())
    fit = SimpleNamespace(
        terminal_rank=rank,
        coefficients=beta,
        coefficient_face=None,
        terminal_curvature=SimpleNamespace(certified=True),
        terminal_data_curvature=data,
        terminal_penalized_curvature=h,
    )
    inverse = rank.pseudo_inverse()
    raw = wood_fasiolo_update((component,), beta, inverse)
    update, names = _stable_isolated_gfs_update(
        (component,), fit, inverse, raw, DistributionalEFSConfig()
    )
    assert names == {"s"}
    expected = np.log(0.5 * lam / (lam + 0.5))
    assert update.raw_log_steps["s"] == pytest.approx(expected, abs=32 * np.finfo(float).eps * lam)


@pytest.mark.pins_behaviour
def test_an_unresolved_reduced_trace_keeps_the_generic_step():
    """Pin refusal when the positive trace deficit is itself at rounding scale."""
    data = np.array([[2.0**-50, 0, 0], [0, 1, 1], [0, 1, 1]])
    h = data + np.diag([1, 0, 0])
    rank = _certified_rank(h, h)
    beta = np.array([0.5, 0, 0])
    fit = SimpleNamespace(
        terminal_rank=rank,
        terminal_curvature=SimpleNamespace(certified=True),
        coefficient_face=None,
        coefficients=beta,
        terminal_penalized_curvature=h,
        terminal_data_curvature=data,
    )
    component = EFSComponentState("s", slice(0, 1), np.eye(1), 1, 1, LambdaPolicy.estimate())
    inverse = rank.pseudo_inverse()
    raw = wood_fasiolo_update((component,), beta, inverse)
    update, names = _stable_isolated_gfs_update(
        (component,), fit, inverse, raw, DistributionalEFSConfig()
    )
    assert not names
    assert update is raw


@pytest.mark.pins_behaviour
@pytest.mark.parametrize(
    "case", ["overlap", "face", "uncertified", "unrepresented", "dropped_penalty"]
)
def test_only_a_certified_retained_isolated_block_can_override_the_generic_step(case):
    """Pin the generic update on a face, shared penalty or unsupported representative."""
    component, fit = _saturated_alias()
    inverse = fit.terminal_rank.pseudo_inverse()
    components = (component,)
    if case == "overlap":
        other = np.eye(2) - component.penalty
        fit.terminal_data_curvature[:2, :2] -= other
        components += (
            replace(
                component,
                name="other",
                penalty=other,
                lambda_value=1.0,
                policy=LambdaPolicy.fixed(1.0),
            ),
        )
    elif case == "face":
        fit.coefficient_face = object()
    elif case == "uncertified":
        fit.terminal_curvature.certified = False
    elif case == "unrepresented":
        fit.terminal_rank = replace(fit.terminal_rank, structural_aliases=None)
    else:
        # Include a dropped nuisance coordinate with zero penalty weight.
        # The component's declared coefficient block now crosses the drop.
        components = (
            replace(
                component, coefficient_slice=slice(0, 4), penalty=np.pad(component.penalty, (0, 2))
            ),
        )
    raw = wood_fasiolo_update(components, fit.coefficients, inverse)
    update, names = _stable_isolated_gfs_update(
        components, fit, inverse, raw, DistributionalEFSConfig()
    )
    assert not names
    assert update is raw
