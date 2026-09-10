"""Only authenticated raw evidence survives a new SSP coordinate map."""

import copy
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from superglm.reml import multi_penalty as kernel
from superglm.reml import penalty_algebra as algebra
from superglm.reml import penalty_support as support_module


def _inputs():
    roots = [np.array([[1.0, 0.0, 0.0]]), np.array([[1.0, 1.0, 0.0]]), np.array([[0.0, 1.0, 0.0]])]
    raw = [root.T @ root for root in roots]
    target_map = np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0], [5.0, 0.0, 6.0]])
    return raw, target_map


def _build(raw, coordinate_map, source=None, *, suffixes=("a", "b", "c"), **metadata):
    group = SimpleNamespace(name="shared", sl=slice(0, 3), size=3)
    matrix = SimpleNamespace(
        R_inv=coordinate_map,
        omega=sum(raw),
        omega_components=list(zip(suffixes, raw, strict=True)),
        **metadata,
    )
    kwargs = {} if source is None else {"_reuse_raw_from": source}
    result = algebra.build_penalty_context([matrix], [(0, group)], **kwargs)
    return result[0]


def _values(values=(2.0, 3.0, 5.0)):
    return dict(zip(("shared:a", "shared:b", "shared:c"), values, strict=True))


def _evaluate(components, values=(2.0, 3.0, 5.0)):
    return algebra._compute_penalty_logdet_evaluation(_values(values), components)


def test_new_map_reuses_raw_support_and_summary_but_rebuilds_map_evidence(monkeypatch):
    raw, target_map = _inputs()
    counts = {"support": 0, "summary": 0, "transport": 0, "volume": 0}
    for module, name, key in [
        (support_module, "_penalty_support", "support"),
        (kernel, "_evaluate_penalty_summary", "summary"),
        (algebra, "_ssp_component_roots", "transport"),
        (algebra, "_support_coordinate_volume", "volume"),
    ]:
        original = getattr(module, name)

        def counted(*args, _original=original, _key=key, **kwargs):
            counts[_key] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, name, counted)
    source = _build(raw, np.eye(3))
    before = _evaluate(source)
    old = algebra._context_geometry(source)
    target = _build(raw, target_map, source)
    current = algebra._context_geometry(target)
    after = _evaluate(target)
    assert current is not old
    assert current.support is old.support
    assert current.last_evaluation[0] is old.last_evaluation[0]
    assert current.coordinate_map is not old.coordinate_map
    assert current.ssp_roots is not old.ssp_roots
    assert current.ssp_root_errors is not old.ssp_root_errors
    assert current.matrix_error_bounds is not old.matrix_error_bounds
    assert current.volume != old.volume
    assert before.rank == after.rank == 2
    assert before.gradient == after.gradient
    assert before.hessian == after.hessian
    assert counts == {"support": 1, "summary": 1, "transport": 2, "volume": 2}
    independent = _evaluate(_build(raw, target_map))
    assert after == independent


def test_changed_weights_reuse_support_and_require_a_fresh_raw_summary(monkeypatch):
    raw, target_map = _inputs()
    calls = []
    original = kernel._evaluate_penalty_summary

    def counted(*args, **kwargs):
        calls.append(tuple(args[1]))
        return original(*args, **kwargs)

    monkeypatch.setattr(kernel, "_evaluate_penalty_summary", counted)
    source = _build(raw, np.eye(3))
    _evaluate(source)
    target = _build(raw, target_map, source)
    after = _evaluate(target, (3.0, 2.0, 5.0))
    assert algebra._context_geometry(target).support is algebra._context_geometry(source).support
    assert calls == [(2.0, 3.0, 5.0), (3.0, 2.0, 5.0)]
    assert after == _evaluate(_build(raw, target_map), (3.0, 2.0, 5.0))


@pytest.mark.parametrize("source_values", [(2.0, 3.0, 5.0), (2.0, 0.0, 5.0), (0.0, 0.0, 0.0)])
@pytest.mark.parametrize("target_values", [(2.0, 3.0, 5.0), (2.0, 0.0, 5.0), (0.0, 0.0, 0.0)])
def test_activity_changes_never_transfer_an_ssp_face(source_values, target_values):
    raw, target_map = _inputs()
    source = _build(raw, np.eye(3))
    _evaluate(source, source_values)
    old = algebra._context_geometry(source)
    target = _build(raw, target_map, source)
    current = algebra._context_geometry(target)
    assert current.support is old.support
    assert current.face_support is None
    if all(source_values):
        assert current.last_evaluation[0] is old.last_evaluation[0]
        assert current.last_evaluation[0]._support is old.support
    else:
        assert current.last_evaluation is None
    after = _evaluate(target, target_values)
    if any(target_values) and not all(target_values):
        assert current.face_support is not None
        assert current.face_support is not old.face_support
    else:
        assert current.face_support is None
    assert after == _evaluate(_build(raw, target_map), target_values)


@pytest.mark.parametrize("change", ["raw", "dtype", "order", "metadata", "incomplete"])
def test_changed_raw_family_or_source_metadata_constructs_a_fresh_support(change):
    raw, target_map = _inputs()
    source = _build(raw, np.eye(3))
    _evaluate(source)
    old = algebra._context_geometry(source)
    options = {}
    if change == "raw":
        # Both the GM input and source omega_raw refer to this array.
        raw[0] *= 2
    elif change == "dtype":
        raw = [matrix.astype(np.float32) for matrix in raw]
    elif change == "order":
        source = source[::-1]
    elif change == "metadata":
        options["component_types"] = {"a": "selection"}
    else:
        source = source[:-1]
    target = _build(raw, target_map, source, **options)
    assert algebra._context_geometry(target).support is not old.support
    assert _evaluate(target) == _evaluate(_build(raw, target_map, **options))


@pytest.mark.parametrize(
    "field",
    [
        "component_roots",
        "Q_plus",
        "Q_zero",
        "component_reconstruction_bounds",
        "component_root_error_bounds",
        "balanced_coordinates",
        "root_log_scales",
        "coordinate_map",
        "coordinate_triangular",
        "support_projection_bounds",
    ],
)
def test_mutated_static_support_evidence_cannot_cross_the_handoff(field):
    raw, target_map = _inputs()
    source = _build(raw, np.eye(3))
    _evaluate(source)
    old = algebra._context_geometry(source)
    value = getattr(old.support, field)
    array = value[0] if isinstance(value, tuple) else value
    assert array.size
    array.setflags(write=True)
    array.flat[0] += 0.125
    array.setflags(write=False)
    target = _build(raw, target_map, source)
    assert algebra._context_geometry(target).support is not old.support
    assert _evaluate(target) == _evaluate(_build(raw, target_map))


@pytest.mark.parametrize("field", ["gradient", "hessian", "gradient_error", "hessian_error"])
def test_mutated_admitted_summary_cannot_be_seeded(field):
    raw, target_map = _inputs()
    source = _build(raw, np.eye(3))
    _evaluate(source)
    old = algebra._context_geometry(source)
    summary = old.last_evaluation[0]
    parent = summary._certificate if field.endswith("_error") else summary
    array = getattr(parent, field)
    array.setflags(write=True)
    array.flat[0] += 0.125
    array.setflags(write=False)
    target = _build(raw, target_map, source)
    current = algebra._context_geometry(target)
    assert current.support is old.support
    assert current.last_evaluation is None
    assert _evaluate(target) == _evaluate(_build(raw, target_map))


@pytest.mark.parametrize("helper", ["_reference_root_actions", "_dyadic_slices"])
def test_changed_arithmetic_token_does_not_reuse_old_evidence(monkeypatch, helper):
    raw, target_map = _inputs()
    source = _build(raw, np.eye(3))
    _evaluate(source)
    original = getattr(kernel, helper, None)

    def replacement(*args, **kwargs):
        return None if original is None else original(*args, **kwargs)

    monkeypatch.setattr(kernel, helper, replacement, raising=False)
    target = _build(raw, target_map, source)
    assert (
        algebra._context_geometry(target).support is not algebra._context_geometry(source).support
    )
    assert _evaluate(target) == _evaluate(_build(raw, target_map))


def test_failed_new_map_does_not_change_the_source_admitted_result():
    raw, _ = _inputs()
    source = _build(raw, np.eye(3))
    expected = _evaluate(source)
    old = algebra._context_geometry(source)
    previous = old.last_evaluation
    with pytest.raises(support_module.PenaltyNumericalError):
        _build(raw, np.zeros((3, 3)), source)
    assert old.last_evaluation is previous
    assert _evaluate(source) == expected


@pytest.mark.parametrize("method", ["pickle", "deepcopy"])
def test_fit_local_receipts_are_dropped_when_the_context_is_copied(method):
    raw, target_map = _inputs()
    source = _build(raw, np.eye(3))
    expected = _evaluate(source)
    original = algebra._context_geometry(source)
    copied = pickle.loads(pickle.dumps(source)) if method == "pickle" else copy.deepcopy(source)
    restored = copied[0]._penalty_geometry
    assert restored.raw_family is None
    assert restored.raw_summary is None
    assert algebra._raw_evidence_value(algebra._raw_support_values(restored.support)) == (
        algebra._raw_evidence_value(algebra._raw_support_values(original.support))
    )
    assert algebra._raw_evidence_value(
        algebra._raw_summary_values(restored.last_evaluation[0])
    ) == (algebra._raw_evidence_value(algebra._raw_summary_values(original.last_evaluation[0])))
    actual = _evaluate(copied)
    assert actual.rank == expected.rank
    assert abs(actual.logdet - expected.logdet) <= actual.logdet_error + expected.logdet_error
    target = _build(raw, target_map, copied)
    assert algebra._context_geometry(target).support is not restored.support
    assert _evaluate(target) == _evaluate(_build(raw, target_map))


def test_finalization_passes_the_optimizer_family_to_the_new_ssp_context(monkeypatch):
    from superglm.model import reml_finalize

    raw, target_map = _inputs()
    source = _build(raw, np.eye(3))
    _evaluate(source)
    old = algebra._context_geometry(source)
    group = SimpleNamespace(name="shared", sl=slice(0, 3), size=3)
    target_matrix = SimpleNamespace(
        R_inv=target_map,
        omega=sum(raw),
        omega_components=list(zip(("a", "b", "c"), raw, strict=True)),
    )
    model = SimpleNamespace(
        _dm=SimpleNamespace(group_matrices=[object()]),
        _groups=[group],
        _discrete=False,
        _direct_solve="auto",
        _distribution=None,
        _link=None,
    )
    mode = SimpleNamespace(beta=np.zeros(3), intercept=0.0)
    best = SimpleNamespace(
        pirls_result=mode,
        lambdas=_values(),
        n_reml_iter=1,
        converged=True,
        curvature_source="fisher",
    )
    monkeypatch.setattr(
        reml_finalize,
        "rebuild_dm_with_lambdas",
        lambda *_args: SimpleNamespace(group_matrices=[target_matrix]),
    )
    monkeypatch.setattr(reml_finalize, "_map_beta_between_bases", lambda beta, *_args: beta)
    monkeypatch.setattr(reml_finalize, "model_weight_semantics", lambda _model: "frequency")

    class ReachedTerminalCoefficientFitError(Exception):
        pass

    def terminal_fit(**kwargs):
        current = algebra._context_geometry(kwargs["reml_penalties"])
        assert current is not old
        assert current.support is old.support
        assert current.last_evaluation[0] is old.last_evaluation[0]
        np.testing.assert_array_equal(current.coordinate_map, target_map)
        raise ReachedTerminalCoefficientFitError

    monkeypatch.setattr(reml_finalize, "fit_irls_direct", terminal_fit)
    with pytest.raises(ReachedTerminalCoefficientFitError):
        reml_finalize.finalize_reml_fit(
            model,
            best=best,
            use_direct=True,
            reml_groups=[(0, group)],
            reml_penalties=source,
            y=np.zeros(4),
            sample_weight=np.ones(4),
            offset=None,
            offset_arr=np.zeros(4),
            max_pirls_iter=10,
            pirls_tol=1e-9,
            qp_passthrough=False,
            qp_saved_state=None,
            profile={},
            total_start=0,
            compute_fit_stats=lambda *_args: None,
        )
