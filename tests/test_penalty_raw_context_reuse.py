"""A raw penalty support survives a new SSP coordinate map; the map's pieces do not."""

from types import SimpleNamespace

import numpy as np
import pytest

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
    return algebra.build_penalty_context([matrix], [(0, group)], **kwargs)[0]


def _values(values=(2.0, 3.0, 5.0)):
    return dict(zip(("shared:a", "shared:b", "shared:c"), values, strict=True))


def _evaluate(components, values=(2.0, 3.0, 5.0)):
    return algebra.penalty_logdet(_values(values), components)


def test_new_map_reuses_the_raw_support_and_rebuilds_the_map_pieces(monkeypatch):
    raw, target_map = _inputs()
    counts = {"support": 0, "transport": 0}
    for module, name, key in [
        (support_module, "_penalty_support", "support"),
        (algebra, "_transported_roots", "transport"),
    ]:
        original = getattr(module, name)

        def counted(*args, _original=original, _key=key, **kwargs):
            counts[_key] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, name, counted)
    source = _build(raw, np.diag([1.0, 2.0, 4.0]))
    before = _evaluate(source)
    old = algebra._context_geometry(source)
    target = _build(raw, target_map, source)
    current = algebra._context_geometry(target)
    after = _evaluate(target)
    assert current is not old and current.support is old.support
    assert current.plain.roots is old.plain.roots
    assert current.plain.volume != old.plain.volume
    assert before.rank == after.rank == 2
    assert before.gradient == after.gradient
    assert before.hessian == after.hessian
    assert counts == {"support": 1, "transport": 2}
    assert after == _evaluate(_build(raw, target_map))


@pytest.mark.parametrize("change", ["raw", "mutated", "order", "incomplete"])
def test_changed_raw_penalties_or_source_construct_a_fresh_support(change):
    raw, target_map = _inputs()
    source = _build(raw, np.diag([1.0, 2.0, 4.0]))
    old = algebra._context_geometry(source)
    if change == "raw":
        raw = [2 * raw[0], *raw[1:]]
    elif change == "mutated":
        # The GM's own arrays, changed in place after the source was built.
        raw[0] *= 2
    elif change == "order":
        source = source[::-1]
    else:
        source = source[:-1]
    target = _build(raw, target_map, source)
    assert algebra._context_geometry(target).support is not old.support
    assert _evaluate(target) == _evaluate(_build(raw, target_map))


@pytest.mark.parametrize("source_values", [(2.0, 3.0, 5.0), (2.0, 0.0, 5.0)])
@pytest.mark.parametrize("target_values", [(2.0, 3.0, 5.0), (2.0, 0.0, 5.0), (0.0, 0.0, 0.0)])
def test_a_zero_weight_face_follows_its_own_map(source_values, target_values):
    raw, target_map = _inputs()
    source = _build(raw, np.diag([1.0, 2.0, 4.0]))
    _evaluate(source, source_values)
    target = _build(raw, target_map, source)
    current = algebra._context_geometry(target)
    assert current.plain.faces == {} and current.plain.last is None
    assert _evaluate(target, target_values) == _evaluate(_build(raw, target_map), target_values)


def test_finalization_passes_the_optimizer_support_to_the_new_ssp_context(monkeypatch):
    from superglm.model import reml_finalize

    raw, target_map = _inputs()
    source = _build(raw, np.diag([1.0, 2.0, 4.0]))
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
    # The optimizer's PIRLSResult, as far as finalize reads it.
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
        lambda *_args: SimpleNamespace(group_matrices=[target_matrix], p=3),
    )
    monkeypatch.setattr(reml_finalize, "_map_beta_between_bases", lambda beta, *_args: beta)
    monkeypatch.setattr(reml_finalize, "model_weight_semantics", lambda _model: "frequency")

    class ReachedTerminalCoefficientFitError(Exception):
        pass

    def terminal_fit(**kwargs):
        current = algebra._context_geometry(kwargs["reml_penalties"])
        assert current is not old
        assert current.support is old.support
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
