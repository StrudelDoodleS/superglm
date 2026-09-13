"""An unchanged discrete tensor retains its selected support within one fit."""

import copy
import gc
import pickle
import weakref
from dataclasses import replace
from decimal import Decimal, localcontext
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.features import Spline
from superglm.group_matrix import DiscretizedTensorGroupMatrix
from superglm.reml import penalty_algebra as algebra
from superglm.reml import penalty_support as support_module
from superglm.types import LambdaPolicy


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
    kwargs = {} if source is None else {"_reuse_fixed_from": source}
    return algebra.build_penalty_context([gm], [(0, group)], **kwargs)[0]


def _evaluate(components, weights=(2.0, 3.0)):
    return algebra._compute_penalty_logdet_evaluation(
        dict(zip(("tensor:left", "tensor:right"), weights, strict=True)), components
    )


def test_fixed_tensor_support_is_lazy_and_shared_with_a_new_owner(monkeypatch):
    gm, group = _tensor_inputs()
    calls = []
    original = support_module._penalty_support

    def counted(matrices):
        calls.append(tuple(matrix.shape for matrix in matrices))
        return original(matrices)

    monkeypatch.setattr(support_module, "_penalty_support", counted)
    source = _build(gm, group)
    old = algebra._context_geometry(source)
    assert old.support is None
    assert calls == []
    _evaluate(source)
    previous = old.last_evaluation
    target = _build(gm, group, source)
    current = algebra._context_geometry(target)
    assert current is not old
    assert current.support is old.support
    assert current.last_evaluation is current.last_weights is None
    assert current.face_support is current.face_activity is None
    assert current.volume is current.volume_activity is None
    assert all(left is not right for left, right in zip(source, target, strict=True))
    assert all(left is right for left, right in zip(old.matrices, current.matrices, strict=True))
    _evaluate(target, (3.0, 2.0))
    assert old.last_evaluation is previous
    assert calls == [((4, 4), (4, 4))]


def test_fixed_tensor_transfer_consumes_the_handoff_receipt():
    """Final model state must not retain duplicate authorization snapshots."""
    gm, group = _tensor_inputs()
    source = _build(gm, group)
    _evaluate(source)
    target = _build(gm, group, source)
    current = algebra._context_geometry(target)
    assert current.fixed_inputs is None
    assert current.fixed_family is None
    next_target = _build(gm, group, target)
    _evaluate(next_target)
    assert algebra._context_geometry(next_target).support is not current.support


@pytest.mark.parametrize("weights,rank", [((2.0, 3.0), 3), ((0.0, 3.0), 2), ((0.0, 0.0), 0)])
def test_fixed_tensor_handoff_preserves_selected_and_active_rank(weights, rank):
    gm, group = _tensor_inputs()
    source = _build(gm, group)
    _evaluate(source)
    target = _build(gm, group, source)
    actual = _evaluate(target, weights)
    expected = _evaluate(_build(gm, group), weights)
    assert actual.rank == expected.rank == rank
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
    assert abs(actual.logdet - expected.logdet) <= actual.logdet_error + expected.logdet_error
    for name in actual.gradient:
        assert abs(actual.gradient[name] - expected.gradient[name]) <= (
            actual.gradient_error[name] + expected.gradient_error[name]
        )


def test_reused_tensor_retains_the_analytic_determinant_and_reconstruction_bounds():
    gm, group = _tensor_inputs()
    source = _build(gm, group)
    _evaluate(source)
    target = _build(gm, group, source)
    current = algebra._context_geometry(target)
    actual = _evaluate(target)
    unit = np.finfo(float).eps
    width = group.size
    with localcontext() as context:
        context.prec = 80
        expected_logdet = float(Decimal(30).ln())
        expected_gradient = [float(Decimal(7) / 5), float(Decimal(8) / 5)]
        expected_curvature = float(Decimal(6) / 25)
    assert actual.rank == 3
    assert abs(actual.logdet - expected_logdet) <= (
        actual.logdet_error + 2 * unit * abs(expected_logdet)
    )
    for name, expected in zip(("tensor:left", "tensor:right"), expected_gradient, strict=True):
        assert (
            abs(actual.gradient[name] - expected)
            <= actual.gradient_error[name] + 2 * unit * expected
        )
    for pair, value in actual.hessian.items():
        expected = expected_curvature if pair[0] == pair[1] else -expected_curvature
        assert abs(value - expected) <= actual.hessian_error[pair] + 2 * unit * abs(expected)
    support = current.support
    for matrix, root, gram_bound, projection_bound in zip(
        current.matrices,
        support.component_roots,
        support.component_reconstruction_bounds,
        support.support_projection_bounds,
        strict=True,
    ):
        # This unit-diagonal fixture has unit equilibration scales. Charge the
        # selected-root displacement separately from its Gram reconstruction.
        root_size = np.linalg.norm(root, ord="fro")
        displacement = np.linalg.norm(projection_bound, ord="fro")
        allowance = (
            np.linalg.norm(gram_bound, ord="fro")
            + 2 * root_size * displacement
            + displacement**2
            + 4 * width * unit * root_size**2
        )
        assert np.linalg.norm(root.T @ root - matrix, ord="fro") <= allowance
    projector = support.Q_plus @ support.Q_plus.T
    assert np.linalg.norm(projector - np.diag([0.0, 1.0, 1.0, 1.0]), ord="fro") <= (
        8 * width * unit
    )


def _change_readonly(array):
    backing = array
    while isinstance(backing.base, np.ndarray):
        backing = backing.base
    backing.setflags(write=True)
    array.setflags(write=True)
    array.flat[-1] += 0.125
    array.setflags(write=False)
    backing.setflags(write=False)


@pytest.mark.parametrize(
    "change",
    [
        "raw",
        "raw_dtype",
        "raw_layout",
        "solver",
        "solver_dtype",
        "solver_layout",
        "source_type",
        "target_type",
        "source_policy",
        "target_policy",
        "rank",
        "logdet",
        "spectrum",
        "order",
        "incomplete",
        "placement",
        "map",
        "map_dtype",
        "map_layout",
        "basis_identity",
        "projection",
    ],
)
@pytest.mark.filterwarnings("ignore:Setting the strides on a NumPy array.*:DeprecationWarning")
def test_changed_fixed_tensor_inputs_refuse_support_transfer(change):
    gm, group = _tensor_inputs()
    source = _build(gm, group)
    _evaluate(source)
    old = algebra._context_geometry(source)
    if change == "raw":
        gm.omega_components[0][1][:] *= 2
    elif change == "raw_dtype":
        gm.omega_components = [
            (name, matrix.astype(np.float32)) for name, matrix in gm.omega_components
        ]
    elif change == "raw_layout":
        gm.omega_components = [
            (name, np.asfortranarray(matrix)) for name, matrix in gm.omega_components
        ]
    elif change == "solver":
        _change_readonly(source[0].omega_ssp)
    elif change == "solver_dtype":
        source[0].omega_ssp = source[0].omega_ssp.astype(np.float32)
    elif change == "solver_layout":
        source[0].omega_ssp.strides = source[0].omega_ssp.strides[::-1]
    elif change == "source_type":
        source[0].component_type = "selection"
    elif change == "target_type":
        gm.component_types["left"] = "selection"
    elif change == "source_policy":
        source[0].lambda_policy = LambdaPolicy.fixed(2.0)
    elif change == "target_policy":
        gm.lambda_policies["left"] = LambdaPolicy.fixed(2.0)
    elif change == "rank":
        source[0].rank = 1.0
    elif change == "logdet":
        source[0].log_det_omega_plus += 1.0
    elif change == "spectrum":
        _change_readonly(source[0].eigvals_omega)
    elif change == "order":
        source = source[::-1]
    elif change == "incomplete":
        source = source[:-1]
    elif change == "placement":
        group.start, group.end, group.sl = 1, 5, slice(1, 5)
    elif change == "map":
        gm.R_inv[0, 1] = 0.125
    elif change == "map_dtype":
        gm.R_inv = gm.R_inv.astype(np.float32)
    elif change == "map_layout":
        gm.R_inv = np.asfortranarray(gm.R_inv)
    elif change == "basis_identity":
        gm.B1_unique_t = gm.B1_unique_t.copy()
    else:
        gm.projection = np.eye(4)
    target = _build(gm, group, source)
    assert algebra._context_geometry(target).support is not old.support
    assert _evaluate(target) == _evaluate(_build(gm, group))


@pytest.mark.parametrize(
    "field",
    [
        "component_roots",
        "balanced_coordinates",
        "root_log_scales",
        "coordinate_map",
        "coordinate_triangular",
        "Q_plus",
        "Q_zero",
        "component_reconstruction_bounds",
        "support_projection_bounds",
        "component_root_error_bounds",
    ],
)
def test_mutated_fixed_tensor_support_evidence_refuses_transfer(field):
    gm, group = _tensor_inputs()
    source = _build(gm, group)
    _evaluate(source)
    old = algebra._context_geometry(source)
    value = getattr(old.support, field)
    array = value[0] if isinstance(value, tuple) else value
    _change_readonly(array)
    target = _build(gm, group, source)
    assert algebra._context_geometry(target).support is not old.support
    assert _evaluate(target) == _evaluate(_build(gm, group))


@pytest.mark.parametrize(
    "field", ["component_roots", "Q_plus", "eigvals_omega", "solver_backing", "spectrum_backing"]
)
def test_writable_fixed_tensor_evidence_refuses_transfer(field):
    gm, group = _tensor_inputs()
    source = _build(gm, group)
    _evaluate(source)
    old = algebra._context_geometry(source)
    if field in {"eigvals_omega", "spectrum_backing"}:
        value = source[0].eigvals_omega
    elif field == "solver_backing":
        value = source[0].omega_ssp
    else:
        value = getattr(old.support, field)
    array = value[0] if isinstance(value, tuple) else value
    backing = array
    while isinstance(backing.base, np.ndarray):
        backing = backing.base
    backing.setflags(write=True)
    if not field.endswith("_backing"):
        array.setflags(write=True)
    target = _build(gm, group, source)
    assert algebra._context_geometry(target).support is not old.support


def test_readonly_support_views_over_a_mutable_buffer_refuse_transfer():
    gm, group = _tensor_inputs()
    source = _build(gm, group)
    _evaluate(source)
    old = algebra._context_geometry(source)
    original = old.support.Q_plus
    replacement = np.ndarray(
        original.shape,
        dtype=original.dtype,
        strides=original.strides,
        buffer=bytearray(original.tobytes(order="A")),
    )
    replacement.setflags(write=False)
    object.__setattr__(old.support, "Q_plus", replacement)
    target = _build(gm, group, source)
    assert algebra._context_geometry(target).support is not old.support


@pytest.mark.parametrize("when", ["before_support", "after_support"])
def test_changed_fixed_tensor_arithmetic_is_not_blessed_by_late_capture(monkeypatch, when):
    gm, group = _tensor_inputs()
    source = _build(gm, group)
    old = algebra._context_geometry(source)
    original = support_module._penalty_support
    if when == "after_support":
        _evaluate(source)

    def replacement(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(support_module, "_penalty_support", replacement)
    if when == "before_support":
        _evaluate(source)
        monkeypatch.setattr(support_module, "_penalty_support", original)
    target = _build(gm, group, source)
    assert algebra._context_geometry(target).support is not old.support


@pytest.mark.parametrize(
    "change", ["precision", "rank_policy", "resolution_flag", "support_layout"]
)
@pytest.mark.filterwarnings("ignore:Setting the strides on a NumPy array.*:DeprecationWarning")
def test_fixed_tensor_precision_and_selected_support_policy_are_authenticated(monkeypatch, change):
    from superglm.solvers import rank

    gm, group = _tensor_inputs()
    source = _build(gm, group)
    _evaluate(source)
    old = algebra._context_geometry(source)
    if change == "precision":
        monkeypatch.setattr(support_module, "_EPS", support_module._EPS * 2)
    elif change == "rank_policy":
        monkeypatch.setattr(
            rank,
            "SHARED_RANK_POLICY",
            replace(rank.SHARED_RANK_POLICY, version=rank.SHARED_RANK_POLICY.version + 1),
        )
    elif change == "resolution_flag":
        flags = old.support.component_resolution_limited
        object.__setattr__(old.support, "component_resolution_limited", (not flags[0], *flags[1:]))
    else:
        old.support.coordinate_triangular.strides = old.support.coordinate_triangular.strides[::-1]
    target = _build(gm, group, source)
    assert algebra._context_geometry(target).support is not old.support


def test_observation_rows_do_not_invalidate_fixed_penalty_support():
    gm, group = _tensor_inputs()
    source = _build(gm, group)
    expected = _evaluate(source)
    old = algebra._context_geometry(source)
    # Observation-side changes require fresh data geometry elsewhere. They do
    # not change this fixed penalty/map target or its marginal column identity.
    gm.B_unique[:] *= 2
    gm.B1_unique_t[:] *= 2
    gm.idx1[:] = gm.idx1[::-1]
    target = _build(gm, group, source)
    assert algebra._context_geometry(target).support is old.support
    assert _evaluate(target) == expected


def test_raw_mutation_before_lazy_support_does_not_rebind_provenance():
    gm, group = _tensor_inputs()
    source = _build(gm, group)
    gm.omega_components[0][1][:] *= 2
    _evaluate(source)
    old = algebra._context_geometry(source)
    target = _build(gm, group, source)
    assert algebra._context_geometry(target).support is not old.support
    assert _evaluate(target) == _evaluate(_build(gm, group))


def test_a_restored_solver_matrix_cannot_authenticate_support_built_during_mutation():
    gm, group = _tensor_inputs()
    source = _build(gm, group)
    old = algebra._context_geometry(source)
    array = source[0].omega_ssp
    _change_readonly(array)
    old.get_support()
    array.base.setflags(write=True)
    array.setflags(write=True)
    array.flat[-1] -= 0.125
    array.setflags(write=False)
    array.base.setflags(write=False)
    target = _build(gm, group, source)
    assert algebra._context_geometry(target).support is not old.support
    assert _evaluate(target) == _evaluate(_build(gm, group))


def test_the_independent_basis_gram_memo_does_not_invalidate_selected_support():
    gm, group = _tensor_inputs()
    source = _build(gm, group)
    expected = _evaluate(source)
    old = algebra._context_geometry(source)
    object.__setattr__(old.support, "_basis_gram_evidence", None)
    target = _build(gm, group, source)
    assert algebra._context_geometry(target).support is old.support
    assert _evaluate(target) == expected


@pytest.mark.parametrize("method", ["pickle", "deepcopy"])
def test_fixed_tensor_receipts_do_not_survive_serialization(method):
    gm, group = _tensor_inputs()
    source = _build(gm, group)
    _evaluate(source)
    copied = pickle.loads(pickle.dumps(source)) if method == "pickle" else copy.deepcopy(source)
    restored = copied[0]._penalty_geometry
    assert restored.fixed_inputs is None
    assert restored.fixed_family is None
    target = _build(gm, group, copied)
    assert algebra._context_geometry(target).support is not restored.support


def test_rejected_fixed_tensor_candidate_preserves_accepted_evaluation():
    gm, group = _tensor_inputs()
    source = _build(gm, group)
    expected = _evaluate(source)
    old = algebra._context_geometry(source)
    previous = old.last_evaluation
    gm.R_inv = np.zeros((4, 4))
    with pytest.raises(ValueError, match="non-positive SSP curvature"):
        _build(gm, group, source)
    assert old.last_evaluation is previous
    assert _evaluate(source) == expected


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
    """Losing the optimizer family or rebuilding final support doubles this count."""
    tensor_widths = []
    original = support_module._penalty_support

    def counted(matrices):
        if len(matrices) == 2:
            tensor_widths.append(matrices[0].shape[0])
        return original(matrices)

    monkeypatch.setattr(support_module, "_penalty_support", counted)
    model = _fit_tensor()
    tensor = next(group for group in model._groups if group.feature_name == "x1:x2")
    assert tensor_widths == [tensor.size]


def test_public_fit_releases_the_optimizer_owner_after_terminal_handoff(monkeypatch):
    """The result must carry populated support without retaining the obsolete owner."""
    from superglm.model import fit_ops

    original = fit_ops.optimize_reml_best
    previous = {}

    def capture(*args, **kwargs):
        best = original(*args, **kwargs)
        assert best.reml_penalties is not None
        entry = [item for item in kwargs["reml_penalties"] if item.group_name == "x1:x2"]
        produced = [item for item in best.reml_penalties if item.group_name == "x1:x2"]
        old = algebra._context_geometry(produced)
        assert old is not algebra._context_geometry(entry)
        assert old.support is not None
        previous["owner"] = weakref.ref(old)
        previous["support"] = weakref.ref(old.support)
        previous["components"] = [weakref.ref(item) for item in produced]
        return best

    monkeypatch.setattr(fit_ops, "optimize_reml_best", capture)
    model = _fit_tensor()
    final = [item for item in model._reml_penalties if item.group_name == "x1:x2"]
    current = algebra._context_geometry(final)
    assert model._reml_result.reml_penalties is model._reml_penalties
    assert current.support is previous["support"]()
    gc.collect()
    assert previous["owner"]() is None
    assert all(component() is None for component in previous["components"])


def test_public_tensor_predictions_match_a_fresh_terminal_support(monkeypatch):
    reused = _fit_tensor()
    monkeypatch.setattr(algebra, "_reuse_fixed_tensor_components", lambda *_args: None)
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
