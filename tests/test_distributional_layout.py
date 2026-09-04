from __future__ import annotations

import dataclasses
from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from superglm._frame import as_eager_frame
from superglm._predictor_compiler import CompiledPredictorDesign
from superglm.distributional.family import ParameterSpec, ParameterSupport
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.predictor import (
    CompiledPredictor,
    Predictor,
    compile_predictors,
)
from superglm.features import Numeric, Spline
from superglm.group_matrix import (
    DesignMatrix,
    FactorSmoothGroupMatrix,
    RandomEffectGroupMatrix,
    SparseSSPGroupMatrix,
)
from superglm.links import IdentityLink
from superglm.model.reml_setup import collect_reml_groups
from superglm.reml.penalty_algebra import (
    build_penalty_components,
    penalty_component_dense_matrix,
)
from superglm.types import GroupSlice, LambdaPolicy, PenaltyComponent

from ._distributional_weights import resolved_prior


def _parameter(name: str, link: str = "identity") -> ParameterSpec:
    return ParameterSpec(name, link, name, ParameterSupport())


def _builds():
    n = 24
    frame = as_eager_frame(
        pd.DataFrame(
            {
                "s(age)": np.linspace(18.0, 75.0, n),
                "region": np.linspace(-1.0, 1.0, n),
            }
        )
    )
    smooth = Spline(
        n_knots=5,
        degree=2,
        penalty="ssp",
        select=True,
        lambda_policy={
            "null": LambdaPolicy.fixed(0.25),
            "wiggle": LambdaPolicy.estimate(),
        },
    )
    return compile_predictors(
        frame,
        resolved_prior(np.ones(n)),
        (_parameter("location"), _parameter("scale", "log")),
        (
            Predictor("location", {"s(age)": smooth, "region": Numeric()}),
            Predictor("scale", {"s(age)": smooth}, link="log", intercept=False),
        ),
    )


def test_stacked_layout_places_intercepts_and_predictor_slopes_in_family_order() -> None:
    builds = _builds()
    layout = build_stacked_layout(builds)
    location, scale = layout.predictors

    assert location.name == "location"
    assert location.coefficient_slice == slice(0, 1 + builds[0].compiled.design.p)
    assert location.intercept_index == 0
    assert scale.coefficient_slice == slice(
        location.coefficient_slice.stop,
        location.coefficient_slice.stop + builds[1].compiled.design.p,
    )
    assert scale.intercept_index is None
    assert layout.n_coefficients == scale.coefficient_slice.stop
    assert layout.coefficient_names[0] == "location:(intercept)"
    assert all(":" in name for name in layout.coefficient_names)


def test_layout_qualifies_terms_and_distinct_double_penalty_components() -> None:
    layout = build_stacked_layout(_builds())

    assert "location:s(age)" in layout.term_slices
    assert "location:region" in layout.term_slices
    assert "scale:s(age)" in layout.term_slices
    assert set(layout.penalty_names) == {
        "location:s(age)#null",
        "location:s(age)#wiggle",
        "scale:s(age)#null",
        "scale:s(age)#wiggle",
    }
    fixed = next(pc for pc in layout.penalties if pc.name == "location:s(age)#null")
    assert fixed.lambda_policy == LambdaPolicy.fixed(0.25)
    assert all(pc.group_name.startswith(("location:", "scale:")) for pc in layout.penalties)
    assert all(not pc.omega_raw.flags.writeable for pc in layout.penalties)

    for state in layout.predictors:
        for component in state.penalties:
            local_group = state.groups[component.group_index]
            assert component.group_sl == local_group.sl
            assert component.group_name == f"{state.name}:{local_group.name}"


def test_penalty_matrix_embeds_only_within_qualified_predictor_blocks() -> None:
    layout = build_stacked_layout(_builds())
    lambdas = {name: float(index + 1) for index, name in enumerate(layout.penalty_names)}
    penalty = layout.penalty_matrix(lambdas)

    assert penalty.shape == (layout.n_coefficients, layout.n_coefficients)
    np.testing.assert_allclose(penalty, penalty.T)
    location, scale = layout.predictors
    assert penalty[location.intercept_index, location.intercept_index] == 0.0
    cross = penalty[location.coefficient_slice, scale.coefficient_slice]
    np.testing.assert_array_equal(cross, np.zeros_like(cross))
    assert np.any(np.diag(penalty[location.coefficient_slice, location.coefficient_slice]) > 0)
    assert np.any(np.diag(penalty[scale.coefficient_slice, scale.coefficient_slice]) > 0)


def test_layout_rejects_unqualified_persisted_names() -> None:
    layout = build_stacked_layout(_builds())

    with pytest.raises(ValueError, match="qualified.*term"):
        replace(layout, term_slices=MappingProxyType({"s(age)": slice(1, 2)}))
    with pytest.raises(ValueError, match="qualified.*penalty"):
        bad = replace(layout.penalties[0], name="s(age)#wiggle")
        replace(layout, penalties=(bad, *layout.penalties[1:]))


def test_layout_rejects_penalties_that_escape_or_mismatch_local_group_blocks() -> None:
    builds = _builds()
    first = builds[0]
    template = first.penalties[0]
    width = first.compiled.design.p
    escaping = replace(template, group_sl=slice(width - 1, width + 1))
    with pytest.raises(ValueError, match="penalty.*outside predictor"):
        build_stacked_layout((replace(first, penalties=(escaping,)), builds[1]))

    mismatched = PenaltyComponent(
        name="region",
        group_name="region",
        group_index=1,
        group_sl=first.penalties[0].group_sl,
        omega_raw=template.omega_raw,
        omega_ssp=template.omega_ssp,
        rank=template.rank,
    )
    with pytest.raises(ValueError, match="penalty.*does not match.*group"):
        build_stacked_layout((replace(first, penalties=(mismatched,)), builds[1]))


# ── PenaltyComponent contract: one gate per penalty_kind ──
#
# ``PenaltyComponent`` carries a COMPACT representation.  For three of master's
# four ``penalty_kind`` values the stored ``omega`` is not the block that lands
# in the global penalty matrix: ``identity`` stores nothing at all, ``repeated``
# and ``sum_to_zero`` store one level block that expands by Kronecker product.
#
# The tests below therefore state the expected block from each kind's OWN
# definition — ``np.eye`` for identity, an explicit ``kron`` against a
# hand-written level Gram for the two expanding kinds, the component's stored
# solver-space matrix for ``dense``.  Comparing the layout against
# ``penalty_component_dense_matrix`` instead would be comparing it against the
# very function it delegates to, which passes for any wrong expansion.

_ROWS = 24
_BLOCK = 3


def _penalty_group_matrix(kind: str, n_levels: int):
    """Return ``(group_matrix, group_width)`` for one master ``penalty_kind``."""
    codes = np.arange(_ROWS, dtype=np.intp) % n_levels
    if kind == "identity":
        return RandomEffectGroupMatrix(codes, n_levels), n_levels
    if kind in ("repeated", "sum_to_zero"):
        basis = sp.csr_matrix(np.eye(_BLOCK)[np.arange(_ROWS) % _BLOCK])
        matrix = FactorSmoothGroupMatrix(
            basis,
            codes,
            n_levels,
            natural_map=np.eye(_BLOCK),
            levels=tuple(range(n_levels)),
            repeated_penalty_components=(("wiggle", np.diag([2.0, 0.8, 0.0])),),
            factor_basis="fs" if kind == "repeated" else "sz",
        )
        return matrix, matrix.coefficient_levels * _BLOCK
    if kind == "dense":
        width = n_levels * _BLOCK
        rng = np.random.default_rng(20260818)
        matrix = SparseSSPGroupMatrix(
            sp.csr_matrix(rng.normal(size=(_ROWS, width))),
            np.triu(np.ones((width, width))) * 0.5 + np.eye(width),
        )
        difference = np.diff(np.eye(width), 2, axis=0)
        matrix.omega = difference.T @ difference
        return matrix, width
    raise AssertionError(f"unhandled penalty kind {kind!r}")


def _single_group_layout(kind: str, n_levels: int):
    """Build a one-predictor layout around one real group of the given kind."""
    group_matrix, width = _penalty_group_matrix(kind, n_levels)
    group = GroupSlice(name="g", start=0, end=width)
    components = build_penalty_components(
        [group_matrix],
        collect_reml_groups([group], [group_matrix]),
    )
    compiled = CompiledPredictorDesign(
        design=DesignMatrix([group_matrix], _ROWS, width),
        groups=(group,),
        specs={},
        feature_order=(),
        interaction_specs={},
        interaction_order=(),
    )
    build = CompiledPredictor(
        name="location",
        parameter_index=0,
        link=IdentityLink(),
        compiled=compiled,
        intercept=False,
        offset=np.zeros(_ROWS),
        penalties=tuple(components),
    )
    return build_stacked_layout((build,)), group_matrix, tuple(components)


# The level penalty ``_penalty_group_matrix`` declares for the two expanding
# kinds.  Its rank is 2 of 3, which is what makes a dropped or misbuilt
# expansion visible as a changed eigenvalue pattern, not only a changed scale.
_LEVEL_PENALTY = np.diag([2.0, 0.8, 0.0])

# ``C.T @ C`` for the sum-to-zero contrast ``C`` that maps ``L - 1`` free level
# blocks onto ``L`` raw blocks summing to zero: the last level is minus the sum
# of the others, so the Gram is ``I + J``.  Written out per level count so the
# expectation cannot re-derive itself from the code under test.
_SUM_TO_ZERO_LEVEL_GRAM = {
    2: np.array([[2.0]]),
    3: np.array([[2.0, 1.0], [1.0, 2.0]]),
}


def _expected_block_from_definition(kind: str, n_levels: int) -> np.ndarray:
    """Return the unweighted block each kind is DEFINED to contribute."""
    if kind == "identity":
        # A random effect penalises every level coefficient at unit strength.
        return np.eye(n_levels)
    if kind == "repeated":
        # ``n_levels`` independent copies of the same level penalty.
        return np.kron(np.eye(n_levels), _LEVEL_PENALTY)
    if kind == "sum_to_zero":
        # The same level penalty seen through the sum-to-zero contrast.
        return np.kron(_SUM_TO_ZERO_LEVEL_GRAM[n_levels], _LEVEL_PENALTY)
    raise AssertionError(f"no independent definition for kind {kind!r}")


@pytest.mark.parametrize("kind", ["identity", "sum_to_zero", "repeated"])
@pytest.mark.parametrize("n_levels", [2, 3])
def test_penalty_matrix_embeds_each_expanding_kind_from_its_own_definition(
    kind: str,
    n_levels: int,
) -> None:
    """The block that lands must match the kind's definition, not its storage.

    ``identity`` stores no matrix, and ``repeated``/``sum_to_zero`` store one
    level block smaller than the coefficient block they penalise, so all three
    would embed the wrong matrix if the layout used ``omega_ssp`` directly, and
    a differently wrong one if the expansion used the wrong Kronecker factor or
    the wrong level Gram.  Each expectation here is written from the kind's
    definition rather than read back from the expander under test.
    """
    layout, _, _ = _single_group_layout(kind, n_levels)
    (component,) = layout.penalties
    assert component.penalty_kind == kind

    lam = 1.5
    actual = layout.penalty_matrix({component.name: lam})
    block = actual[component.group_sl, component.group_sl]

    expected = lam * _expected_block_from_definition(kind, n_levels)
    assert block.shape == expected.shape
    np.testing.assert_allclose(block, expected, atol=1e-12)

    # The single group covers the whole layout, so nothing may land outside it.
    outside = actual.copy()
    outside[component.group_sl, component.group_sl] = 0.0
    np.testing.assert_array_equal(outside, np.zeros_like(outside))


@pytest.mark.parametrize("n_levels", [2, 3])
def test_penalty_matrix_embeds_a_dense_kind_as_its_stored_solver_space_block(
    n_levels: int,
) -> None:
    """``dense`` is the one kind whose stored matrix IS the embedded block."""
    layout, _, _ = _single_group_layout("dense", n_levels)
    (component,) = layout.penalties
    assert component.penalty_kind == "dense"

    lam = 2.25
    block = layout.penalty_matrix({component.name: lam})[
        component.group_sl,
        component.group_sl,
    ]
    stored = np.asarray(component.omega_ssp)
    np.testing.assert_allclose(block, lam * stored)

    # R_inv is not the identity in this fixture, so embedding the RAW omega
    # would have been a different matrix rather than a harmless alias.
    assert not np.allclose(stored, np.asarray(component.omega_raw))


def test_penalty_matrix_refuses_a_component_with_no_solver_space_matrix() -> None:
    """A component that never got an ``omega_ssp`` must fail, not be guessed at.

    Every component ``build_penalty_components`` produces for a predictor
    carries its own solver-space matrix, so the layout deliberately passes no
    group matrix to the expander.  Resolving one by ``group_index`` would have
    matched a LOCAL index against the global sequence and silently penalised
    with another predictor's geometry; refusing outright is the safe failure.
    """
    layout, _, _ = _single_group_layout("dense", 3)
    (component,) = layout.penalties
    raw_only = replace(component, omega_ssp=None)
    raw_layout = replace(layout, penalties=(raw_only,))

    assert raw_only.omega_raw is not None
    with pytest.raises(ValueError, match="no solver-space matrix"):
        raw_layout.penalty_matrix({raw_only.name: 1.0})


@pytest.mark.parametrize("kind", ["identity", "sum_to_zero", "repeated", "dense"])
def test_embedding_carries_every_penalty_component_field(kind: str) -> None:
    layout, _, components = _single_group_layout(kind, 3)
    assert len(layout.penalties) == len(components)

    placement = {"name", "group_name", "group_index", "group_sl"}
    for source, embedded in zip(components, layout.penalties, strict=True):
        for descriptor in dataclasses.fields(PenaltyComponent):
            if descriptor.name in placement:
                continue
            original = getattr(source, descriptor.name)
            carried = getattr(embedded, descriptor.name)
            if isinstance(original, np.ndarray):
                np.testing.assert_array_equal(carried, original)
                assert not carried.flags.writeable
            else:
                assert carried == original


def test_penalty_matrix_expands_a_two_level_sum_to_zero_block_rather_than_halving_it() -> None:
    """The one live shape that was silently penalised at half strength.

    At two levels a sum-to-zero factor smooth has exactly one free block, so
    the stored level-sized omega is square at the group width and every shape
    guard passes.  Embedding it directly therefore lands ``omega`` where
    ``C.T @ C (== 2) x omega`` belongs, and the fit converges reporting
    success on half the penalty it was asked for.

    The component is re-embedded here with ``dataclasses.replace`` so it keeps
    its declared ``penalty_kind``: that isolates the expansion defect from the
    separate defect of the layout's copy dropping the field.
    """
    layout, group_matrix, components = _single_group_layout("sum_to_zero", 2)
    source = components[0]
    assert source.penalty_kind == "sum_to_zero"

    embedded = replace(source, name="location:g#wiggle", group_name="location:g")
    isolated = replace(layout, penalties=(embedded,))
    block = isolated.penalty_matrix({embedded.name: 1.0})[
        embedded.group_sl,
        embedded.group_sl,
    ]

    stored = np.asarray(source.omega_ssp)
    assert stored.shape == (embedded.group_sl.stop - embedded.group_sl.start,) * 2
    np.testing.assert_allclose(block, 2.0 * stored)
    np.testing.assert_allclose(block, penalty_component_dense_matrix(embedded, group_matrix))
