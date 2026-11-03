"""The editor restricts to the face the constrained fit recorded (#541)."""

from itertools import permutations
from types import SimpleNamespace

import numpy as np
import pytest

from superglm.editor import free_levels
from superglm.editor.free_levels import _on_binding_face
from superglm.solvers.irls_direct import _active_face_basis
from superglm.types import GroupSlice


def _restriction(rows, face, inverse, groups=None, active=None):
    width = rows.shape[1]
    if groups is None:
        groups = [GroupSlice(name="t", start=0, end=width)]
    if active is None:
        active = groups
    record = SimpleNamespace(face=face, face_rows=rows)
    model = SimpleNamespace(
        _groups=groups,
        result=SimpleNamespace(beta=np.zeros(width)),
        _solver_pirls_result=lambda: SimpleNamespace(rank_info=record),
    )
    return _on_binding_face(
        model, active, lambda v: inverse @ v, SimpleNamespace(p=inverse.shape[0]), None
    )


def test_many_redundant_rows_restrict_to_the_retained_face():
    """Keeping one original row reintroduces a direction the fit discarded.

    With 512 rows the fit's cutoff discards the 1e-12 slant, leaving its
    average in the retained direction. The old restriction missed the
    recorded-face quadratic response by 9.98e-6 of the unrestricted one.
    """
    rows = np.tile([1.0, 0.0, 0.0], (512, 1))
    rows[0, 1] = 1e-12
    face = _active_face_basis(rows)
    assert face.shape == (3, 2)
    inverse = np.diag([1.0, 1e14, 1.0])
    contrast = np.array([1.0, 1e-7, 0.0])
    # Independent restriction through the retained free coordinates, not
    # the row-space Schur complement used by the implementation.
    expected = (
        contrast
        @ face
        @ np.linalg.solve(face.T @ np.linalg.solve(inverse, face), face.T @ contrast)
    )
    for order in (np.arange(512), np.roll(np.arange(512), 1)):
        move = _restriction(rows[order], face, inverse)
        assert move is not None
        actual = contrast @ move(contrast)
        assert abs(actual - expected) <= 1e-6 * (contrast @ inverse @ contrast)


@pytest.mark.parametrize("selected_columns", [False, True])
def test_saved_face_rank_survives_opening_with_a_different_rank_cut(selected_columns):
    """A saved rank decision is authoritative even if today's SVD would keep more.

    Force the retained rank, rather than depending on a platform's rounding
    at the cutoff. The inactive group, when present, holds another direction
    that must be removed from the rank when selecting the active group.
    """
    rows = np.array([[1.0, 0.0, 0.0], [1.0, 1e-12, 0.0]])
    face = np.eye(3)[:, 1:]
    groups = active = None
    if selected_columns:
        rows = np.block([[np.zeros((2, 2)), rows], [np.array([[0.0, 1.0]]), np.zeros((1, 3))]])
        face = np.eye(5)[:, [0, 3, 4]]
        groups = [GroupSlice(name="off", start=0, end=2), GroupSlice(name="t", start=2, end=5)]
        active = [GroupSlice(name="t", start=0, end=3)]
    move = _restriction(rows, face, np.eye(3), groups, active)
    assert move is not None
    contrast = np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(move(contrast), contrast, rtol=1e-6, atol=1e-12)


def test_tied_rows_keep_the_saved_subspace_in_every_row_order():
    """A strict observable check replaces the old conditional expected failure."""
    d = 2.0**-51
    rows = np.array([[1.0, d, 0.0], [2.0, 0.0, 0.0], [1.0, -d, 0.0]])
    face = np.eye(3)[:, 1:]
    inverse = np.diag([1.0, 2.0**96, 1.0])
    contrast = np.array([0.0, 2.0**-48, 0.0])
    for order in permutations(range(3)):
        move = _restriction(rows[list(order)], face, inverse)
        assert move is not None
        # The saved face holds e0, so e1's variance is exactly 2**96.
        assert contrast @ move(contrast) == pytest.approx(1.0, rel=1e-6)


def test_a_nonfinite_gap_variance_is_disclosed_as_unmeasured(monkeypatch):
    """Force a non-finite response; the gap must say it cannot be measured."""
    from types import SimpleNamespace

    def influence(_):
        return np.array([np.inf, 0.0])

    model = SimpleNamespace(result=SimpleNamespace(phi=1.0))
    monkeypatch.setattr(free_levels, "_term_covariance", lambda *args: np.eye(2))
    monkeypatch.setattr(
        free_levels,
        "_influence",
        lambda *args: free_levels._Influence(
            influence=influence, width=2, fisher=np.ones(2), spread=np.ones(2)
        ),
    )
    variances, note = free_levels._gap_variances(
        model, np.array([[1.0, 0.0]]), model, None, "t", None
    )
    assert variances is None
    assert note == free_levels._UNMEASURED


def test_a_record_with_a_face_but_no_face_rows_reads_its_rows_from_the_constraints():
    """A record from before fits kept ``face_rows`` can still carry ``face``."""
    constraints = SimpleNamespace(n_constraints=1, A=np.array([[1.0, -1.0, 0.0]]), b=np.zeros(1))
    group = SimpleNamespace(name="t", start=0, end=3, constraints=constraints)
    record = SimpleNamespace(face=_active_face_basis(constraints.A), face_rows=None)
    model = SimpleNamespace(
        _groups=[group],
        result=SimpleNamespace(beta=np.zeros(3)),
        _solver_pirls_result=lambda: SimpleNamespace(rank_info=record),
    )
    move = _on_binding_face(
        model, [group], lambda v: v, SimpleNamespace(p=3), None, data_gram=np.eye(3)
    )
    assert move is not None
    # The binding row theta_0 = theta_1 holds: I - r r' / (r' r) on (1, 0, 0).
    np.testing.assert_allclose(move(np.array([1.0, 0.0, 0.0])), [0.5, 0.5, 0.0], atol=1e-12)
