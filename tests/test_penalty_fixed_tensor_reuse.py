"""An unchanged discrete tensor keeps its selected support within one fit."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.features import Spline
from superglm.group_matrix import DiscretizedTensorGroupMatrix
from superglm.reml import penalty_algebra as algebra
from superglm.reml import penalty_support as support_module


def _tensor_inputs():
    marginal = np.eye(2)
    gm = DiscretizedTensorGroupMatrix(
        marginal,
        marginal.copy(),
        np.repeat(np.arange(2), 2),
        np.tile(np.arange(2), 2),
        np.eye(4),
        np.eye(4),
        np.arange(4),
        tensor_id=7,
    )
    gm.omega_components = [
        ("left", np.diag([0.0, 0.0, 1.0, 1.0])),
        ("right", np.diag([0.0, 1.0, 0.0, 1.0])),
    ]
    gm.omega = sum(matrix for _, matrix in gm.omega_components)
    gm.component_types = {"left": "wiggle", "right": "wiggle"}
    gm.lambda_policies = {}
    group = SimpleNamespace(name="tensor", start=0, end=4, sl=slice(0, 4), size=4)
    return gm, group


def _build(gm, group, source=None):
    kwargs = {"cache": {}} if source is None else {"_reuse_raw_from": source}
    return algebra.build_penalty_context([gm], [(0, group)], **kwargs)[0]


def _evaluate(components, weights=(2.0, 3.0)):
    return algebra.penalty_logdet(
        dict(zip(("tensor:left", "tensor:right"), weights, strict=True)), components
    )


def test_fixed_tensor_support_is_lazy_and_follows_unchanged_matrices(monkeypatch):
    gm, group = _tensor_inputs()
    calls = []
    original = support_module._penalty_support

    def counted(matrices):
        calls.append(tuple(matrix.shape for matrix in matrices))
        return original(matrices)

    monkeypatch.setattr(support_module, "_penalty_support", counted)
    source = _build(gm, group)
    old = algebra._context_geometry(source)
    assert old.support is None and calls == []
    expected = _evaluate(source)
    target = _build(gm, group, source)
    current = algebra._context_geometry(target)
    assert current is not old and current.support is old.support
    assert all(left is not right for left, right in zip(source, target, strict=True))
    assert _evaluate(target) == expected
    assert calls == [((4, 4), (4, 4))]


@pytest.mark.parametrize("weights,rank", [((2.0, 3.0), 3), ((0.0, 3.0), 2), ((0.0, 0.0), 0)])
def test_fixed_tensor_handoff_preserves_selected_and_active_rank(weights, rank):
    gm, group = _tensor_inputs()
    source = _build(gm, group)
    _evaluate(source)
    target = _build(gm, group, source)
    actual = _evaluate(target, weights)
    assert actual == _evaluate(_build(gm, group), weights)
    assert actual.rank == rank
    assert algebra.compute_total_penalty_rank(target) == 3
    assert (
        algebra.compute_penalty_nullity(
            hessian_rank=5,
            coefficient_width=4,
            penalties=target,
            lambdas=dict(zip(("tensor:left", "tensor:right"), weights, strict=True)),
        )
        == 5 - rank
    )


def test_changed_fixed_tensor_matrices_select_a_fresh_support():
    gm, group = _tensor_inputs()
    source = _build(gm, group)
    _evaluate(source)
    gm.omega_components = [(name, 2 * matrix) for name, matrix in gm.omega_components]
    target = _build(gm, group, source)
    assert (
        algebra._context_geometry(target).get_support()
        is not algebra._context_geometry(source).support
    )


def _fit_tensor():
    rng = np.random.default_rng(77)
    x1, x2 = rng.uniform(size=(2, 260))
    y = 0.2 + np.sin(2 * np.pi * x1) + 0.3 * np.cos(2 * np.pi * x2) + rng.normal(0, 0.3, 260)
    return SuperGLM(
        family="gaussian",
        selection_penalty=0,
        discrete=True,
        features={"x1": Spline(n_knots=5), "x2": Spline(n_knots=5)},
        interactions=[("x1", "x2")],
    ).fit_reml(pd.DataFrame({"x1": x1, "x2": x2}), y, max_reml_iter=3, reml_tol=1e-12)


def test_public_fit_constructs_the_tensor_support_once(monkeypatch):
    """Rebuilding the terminal context's support doubles this count."""
    tensor_widths = []
    original = support_module._penalty_support

    def counted(matrices, **kwargs):
        if len(matrices) == 2:
            tensor_widths.append(matrices[0].shape[0])
        return original(matrices, **kwargs)

    monkeypatch.setattr(support_module, "_penalty_support", counted)
    model = _fit_tensor()
    tensor = next(group for group in model._groups if group.feature_name == "x1:x2")
    assert tensor_widths == [tensor.size]


def test_public_tensor_predictions_match_a_fresh_terminal_support(monkeypatch):
    reused = _fit_tensor()
    monkeypatch.setattr(algebra, "_reusable_solver_context", lambda *_args: None)
    fresh = _fit_tensor()
    design = np.column_stack([np.ones(reused._dm.shape[0]), reused._dm.toarray()])
    hessian = design.T @ design
    for component in reused._reml_penalties:
        block = slice(component.group_sl.start + 1, component.group_sl.stop + 1)
        hessian[block, block] += reused._reml_lambdas[component.name] * component.omega_ssp
    relative_allowance = 16 * len(hessian) * np.finfo(float).eps * np.linalg.cond(hessian)
    assert relative_allowance < np.sqrt(np.finfo(float).eps)
    points = np.linspace(0.05, 0.95, 40)
    held_out = pd.DataFrame({"x1": points, "x2": np.roll(points, 7)})
    expected = fresh.predict(held_out)
    assert np.linalg.norm(reused.predict(held_out) - expected) <= (
        relative_allowance * np.linalg.norm(expected)
    )
    assert algebra.compute_total_penalty_rank(reused._reml_penalties) == (
        algebra.compute_total_penalty_rank(fresh._reml_penalties)
    )
    assert abs(reused._result.phi - fresh._result.phi) <= (relative_allowance * fresh._result.phi)
