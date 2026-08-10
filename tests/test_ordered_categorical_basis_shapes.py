"""``OrderedCategorical(basis=Piecewise(...) | Polynomial(...))`` hosting.

Pins the 0.25.0 inner-basis contract: band-name resolution (names to level
positions at construction, unknown names loud, integer positions as the
escape hatch), the same vocabulary for ``Spline(knots=[names])``, the
position axis 0..L-1, deep-copy isolation for every basis type, the
collapse-times-breaks refusals with the within-segment round trip, the
two-block ``specials=`` contract, level-axis extrapolation inertness,
interaction refusals at registration and resolution, and the fitted-values
equivalence of the hosted degree-1 Piecewise with the numeric-axis term on
positions.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import OrderedCategorical, Piecewise, Polynomial, Spline, SuperGLM
from superglm.editor.collapse import _ordered_spec_with_grouping
from superglm.features.grouping import collapse_levels
from superglm.features.ordered_categorical import resolve_interaction_parent
from superglm.types import GroupInfo

LEVELS = [f"Mi{i:03d}" for i in range(8)]


def _frame(n: int = 3000, seed: int = 20260810) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"band": rng.choice(LEVELS, n), "x": rng.uniform(0.0, 1.0, n)})
    position = {level: index for index, level in enumerate(LEVELS)}
    signal = np.array([0.08 * min(position[b], 4) for b in X["band"]])
    y = signal + 0.3 * X["x"].to_numpy() + rng.normal(0.0, 0.05, n)
    return X, y


def _grouping(groups: dict[str, list[str]], levels: list[str] | None = None):
    levels = LEVELS if levels is None else levels
    data = np.array(levels * 4, dtype=object)
    full = {label: list(members) for label, members in groups.items()}
    covered = {member for members in full.values() for member in members}
    for level in levels:
        if level not in covered:
            full[level] = [level]
    return collapse_levels(data, groups=full, order=levels)


# ── basis acceptance and name resolution ─────────────────────────────


def test_basis_accepts_all_three_and_names_them_in_the_refusal() -> None:
    OrderedCategorical(order=LEVELS, basis=Piecewise(breaks=["Mi004"]))
    OrderedCategorical(order=LEVELS, basis=Polynomial(powers=[1, 2]))
    with pytest.raises(ValueError, match="Spline, Piecewise or Polynomial"):
        OrderedCategorical(order=LEVELS, basis=object())


def test_break_names_resolve_to_level_positions_at_construction() -> None:
    spec = OrderedCategorical(order=LEVELS, basis=Piecewise(breaks=["Mi002", "Mi005"]))
    assert spec._spline.breaks == [2.0, 5.0]
    assert spec._spline.lower == 0.0
    assert spec._spline.upper == 7.0
    assert spec.basis_kind == "piecewise"
    # The pristine declaration keeps the names for editor round trips.
    assert spec._spline_obj.breaks == ["Mi002", "Mi005"]


def test_unknown_break_name_refuses_listing_the_levels() -> None:
    with pytest.raises(ValueError, match=r"'Mi999'.*Mi000") as excinfo:
        OrderedCategorical(order=LEVELS, basis=Piecewise(breaks=["Mi999"]))
    assert "Mi007" in str(excinfo.value)


def test_integer_positions_are_the_escape_hatch() -> None:
    named = OrderedCategorical(order=LEVELS, basis=Piecewise(breaks=["Mi002", "Mi005"]))
    positional = OrderedCategorical(order=LEVELS, basis=Piecewise(breaks=[2, 5]))
    assert positional._spline.breaks == named._spline.breaks
    # Integer-valued floats resolve too; a break between bands does not exist.
    assert OrderedCategorical(order=LEVELS, basis=Piecewise(breaks=[2.0]))._spline.breaks == [2.0]
    with pytest.raises(ValueError, match="between bands"):
        OrderedCategorical(order=LEVELS, basis=Piecewise(breaks=[2.5]))


def test_boundary_and_misordered_breaks_refuse_with_names() -> None:
    with pytest.raises(ValueError, match="first"):
        OrderedCategorical(order=LEVELS, basis=Piecewise(breaks=["Mi000"]))
    with pytest.raises(ValueError, match="last"):
        OrderedCategorical(order=LEVELS, basis=Piecewise(breaks=["Mi007"]))
    with pytest.raises(ValueError, match="ascending level order"):
        OrderedCategorical(order=LEVELS, basis=Piecewise(breaks=["Mi005", "Mi002"]))


def test_values_sets_the_order_but_positions_stay_0_to_L_minus_1() -> None:
    values = {level: float(10 + 7 * index) for index, level in enumerate(LEVELS)}
    spec = OrderedCategorical(values=values, basis=Piecewise(breaks=["Mi004"]))
    assert spec._level_to_value == {level: float(i) for i, level in enumerate(LEVELS)}
    assert spec._spline.breaks == [4.0]


def test_segment_span_must_carry_the_stated_degree() -> None:
    with pytest.raises(ValueError, match="degree 3, which needs at least 4"):
        OrderedCategorical(
            order=LEVELS, basis=Piecewise(breaks=["Mi002", "Mi004"], degrees=[1, 3, 1])
        )


def test_polynomial_power_ceiling_is_the_level_count() -> None:
    with pytest.raises(ValueError, match="max\\(powers\\) <= "):
        OrderedCategorical(order=LEVELS[:3], basis=Polynomial(powers=[1, 3]))


def test_spline_knots_by_name_resolve_to_level_values() -> None:
    spec = OrderedCategorical(order=LEVELS, basis=Spline(kind="cr", knots=["Mi002", "Mi005"]))
    assert spec.basis_kind == "spline"
    assert np.allclose(spec._spline._explicit_knots, [2 / 7, 5 / 7])
    assert spec._spline.n_knots == 2
    with pytest.raises(ValueError, match=r"'Mi999'"):
        OrderedCategorical(order=LEVELS, basis=Spline(kind="cr", knots=["Mi999"]))


def test_spline_named_knots_refuse_the_numeric_axis() -> None:
    X, y = _frame()
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": Spline(kind="cr", knots=["Mi002"])},
    )
    with pytest.raises(ValueError, match="level names"):
        model.fit(X, y)


# ── deep-copy isolation ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "basis, mutate",
    [
        (
            Piecewise(breaks=["Mi002"], degrees=[1, 2]),
            lambda b: b.__dict__.update(breaks=["Mi005"]),
        ),
        (Polynomial(powers=[1, 2]), lambda b: b.__dict__.update(powers=(1, 2, 3))),
        (Spline(kind="cr", knots=["Mi002"]), lambda b: b.__dict__.update(_named_knots=["Mi005"])),
    ],
)
def test_caller_mutation_after_construction_changes_nothing(basis, mutate) -> None:
    spec = OrderedCategorical(order=LEVELS, basis=basis)
    before_obj = repr(spec._spline_obj.__dict__)
    before_inner = repr(spec._spline.__dict__)
    mutate(basis)
    assert repr(spec._spline_obj.__dict__) == before_obj
    assert repr(spec._spline.__dict__) == before_inner


# ── two-block contract and level-axis policies ───────────────────────


def test_two_block_contract_holds_for_each_inner_basis() -> None:
    X, y = _frame()
    x = X["band"].to_numpy(dtype=object)
    w = np.ones(len(x))
    for basis in (
        Piecewise(breaks=["Mi004"]),
        Polynomial(powers=[1, 2]),
        Spline(kind="cr", n_knots=4),
    ):
        spec = OrderedCategorical(order=LEVELS, basis=basis, specials=["Mi007"])
        infos = spec.build(x, w)
        assert isinstance(infos, list) and len(infos) == 2
        main, special = infos
        assert special.subgroup_name == "special"
        assert special.penalized is False
        assert special.n_cols == 1


def test_parametric_main_blocks_are_unpenalized_and_uncompressed() -> None:
    """Both parametric bases emit an unpenalized main block, symmetrically.

    The hosted Polynomial deliberately diverges from the numeric-axis term's
    group-selection contract: the reported vocabulary (plain whole-term Wald,
    clean per-power z) is invalid the moment lambda1 shrinkage touches the
    block, so the block must sit outside selection exactly as the Piecewise
    and specials blocks do.
    """
    X, _ = _frame()
    x = X["band"].to_numpy(dtype=object)
    for basis in (Piecewise(breaks=["Mi004"]), Polynomial(powers=[1, 2])):
        spec = OrderedCategorical(order=LEVELS, basis=basis)
        info = spec.build(x, np.ones(len(x)))
        assert isinstance(info, GroupInfo)
        assert info.penalized is False
        assert info.supports_row_compression is False


@pytest.mark.parametrize(
    "basis", [Polynomial(powers=[1, 2]), Piecewise(breaks=["Mi004"])], ids=["polynomial", "piecewise"]
)
def test_selection_penalty_leaves_the_hosted_block_bit_identical(basis) -> None:
    """A selection penalty must not shrink a hosted parametric block.

    Observed on the unfixed code for the Polynomial inner: coefficient norm
    0.116 -> 0.093 under selection_penalty=50, silently invalidating every
    clean-z row the summary printed.
    """
    X, y = _frame()

    def fit(penalty: float) -> SuperGLM:
        model = SuperGLM(
            family="gaussian",
            selection_penalty=penalty,
            features={"band": OrderedCategorical(order=LEVELS, basis=basis)},
        )
        model.fit(X, y)
        return model

    unshrunk = fit(0.0)
    shrunk = fit(50.0)
    result_a = getattr(unshrunk, "_result", None) or unshrunk._solver_result
    result_b = getattr(shrunk, "_result", None) or shrunk._solver_result
    assert np.array_equal(np.asarray(result_a.beta), np.asarray(result_b.beta))
    assert np.array_equal(unshrunk.predict(X.head(50)), shrunk.predict(X.head(50)))


def test_extrapolation_parameter_is_inert_on_the_level_axis() -> None:
    X, y = _frame()
    fits = {}
    for mode in ("clip", "error"):
        spec = OrderedCategorical(
            order=LEVELS, basis=Piecewise(breaks=["Mi004"], extrapolation=mode)
        )
        model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"band": spec})
        model.fit(X, y)
        fits[mode] = model.predict(X.head(100))
    assert np.array_equal(fits["clip"], fits["error"])


def test_unseen_level_contract_is_untouched() -> None:
    X, y = _frame()
    spec = OrderedCategorical(order=LEVELS, basis=Piecewise(breaks=["Mi004"]))
    model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"band": spec})
    model.fit(X, y)
    bad = pd.DataFrame({"band": ["NEVER_SEEN"], "x": [0.5]})
    with pytest.raises(ValueError, match="NEVER_SEEN"):
        model.predict(bad)


# ── fitted-values equivalence with the numeric-axis term ─────────────


def test_hosted_degree_one_equals_numeric_piecewise_on_positions() -> None:
    """OC(basis=Piecewise) with all-1 degrees IS today's Piecewise on levels.

    The emitted design is bit-identical to the numeric-axis term built on the
    positions. Predictions agree to float accumulation noise only: the numeric
    term stores its design through the row-compression container, a code path
    the hosted block deliberately does not take.
    """
    X, y = _frame()
    position = {level: float(index) for index, level in enumerate(LEVELS)}
    X_numeric = X.assign(band=[position[b] for b in X["band"]])

    hosted_spec = OrderedCategorical(order=LEVELS, basis=Piecewise(breaks=["Mi002", "Mi005"]))
    numeric_spec = Piecewise(breaks=[2.0, 5.0])
    hosted_info = hosted_spec.build(X["band"].to_numpy(dtype=object), np.ones(len(X)))
    numeric_info = numeric_spec.build(X_numeric["band"].to_numpy(), np.ones(len(X)))
    assert (hosted_info.columns != numeric_info.columns).nnz == 0

    hosted = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "band": OrderedCategorical(order=LEVELS, basis=Piecewise(breaks=["Mi002", "Mi005"]))
        },
    ).fit(X, y)
    numeric = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"band": Piecewise(breaks=[2.0, 5.0])},
    ).fit(X_numeric, y)
    assert np.allclose(hosted.predict(X), numeric.predict(X_numeric), rtol=0.0, atol=1e-12)


def test_stated_all_one_degrees_fit_bit_identically_to_the_default() -> None:
    """``degrees=[1, 1, 1]`` states the default and shares its exact path."""
    X, y = _frame()
    stated = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "band": OrderedCategorical(
                order=LEVELS, basis=Piecewise(breaks=["Mi002", "Mi005"], degrees=[1, 1, 1])
            )
        },
    ).fit(X, y)
    default = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "band": OrderedCategorical(order=LEVELS, basis=Piecewise(breaks=["Mi002", "Mi005"]))
        },
    ).fit(X, y)
    assert np.array_equal(stated.predict(X), default.predict(X))


# ── collapse x breaks ────────────────────────────────────────────────


def test_collapse_absorbing_a_break_refuses_naming_break_and_group() -> None:
    grouping = _grouping({"Mi004+Mi005": ["Mi004", "Mi005"]})
    with pytest.raises(ValueError, match=r"absorbs the stated Piecewise break at level 'Mi004'"):
        OrderedCategorical(order=LEVELS, basis=Piecewise(breaks=["Mi004"]), grouping=grouping)


def test_collapse_straddling_a_break_refuses() -> None:
    grouping = _grouping({"wrap": ["Mi003", "Mi005"]})
    with pytest.raises(ValueError, match="straddles the stated Piecewise break"):
        OrderedCategorical(order=LEVELS, basis=Piecewise(breaks=["Mi004"]), grouping=grouping)


def test_collapse_guard_covers_named_spline_knots() -> None:
    grouping = _grouping({"Mi004+Mi005": ["Mi004", "Mi005"]})
    with pytest.raises(ValueError, match="absorbs the stated Spline knot"):
        OrderedCategorical(
            order=LEVELS, basis=Spline(kind="cr", knots=["Mi004"]), grouping=grouping
        )


def test_within_segment_collapse_re_resolves_the_break() -> None:
    grouping = _grouping({"Mi001+Mi002": ["Mi001", "Mi002"]})
    spec = OrderedCategorical(order=LEVELS, basis=Piecewise(breaks=["Mi004"]), grouping=grouping)
    # Seven grouped levels; the named break follows its level to position 3.
    assert spec._spline.breaks == [3.0]
    assert spec._spline.upper == 6.0


def test_editor_collapse_and_ungroup_round_trip_stays_green() -> None:
    X, y = _frame()
    spec = OrderedCategorical(order=LEVELS, basis=Piecewise(breaks=["Mi004"], degrees=[2, 1]))
    model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"band": spec})
    model.fit(X, y)
    baseline = model.predict(X.head(50))

    data = X["band"].to_numpy(dtype=object)
    grouped = _ordered_spec_with_grouping(
        spec,
        _grouping({"Mi001+Mi002": ["Mi001", "Mi002"]}),
        ["Mi001", "Mi002"],
        "most_exposed",
        data,
    )
    assert grouped._spline.breaks == [3.0]
    collapsed_model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"band": grouped})
    collapsed_model.fit(X, y)

    # Undo through the same editor clone path: an identity grouping becomes
    # grouping=None, and the break re-resolves to the declared position.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ungrouped = _ordered_spec_with_grouping(
            grouped, None, ["Mi001", "Mi002"], "most_exposed", data
        )
    assert ungrouped._spline.breaks == [4.0]
    restored = SuperGLM(family="gaussian", selection_penalty=0.0, features={"band": ungrouped})
    restored.fit(X, y)
    assert np.allclose(restored.predict(X.head(50)), baseline, atol=1e-10)


# ── interaction refusals ─────────────────────────────────────────────


@pytest.mark.parametrize("basis", [Piecewise(breaks=["Mi004"]), Polynomial(powers=[1, 2])])
def test_interactions_refuse_at_registration(basis) -> None:
    spec = OrderedCategorical(order=LEVELS, basis=basis)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"band": spec, "x": Spline(kind="cr", n_knots=4)},
    )
    with pytest.raises(NotImplementedError, match="no marginal smooth to cross with"):
        model._add_interaction("band", "x")


def test_resolve_interaction_parent_refuses_as_backstop() -> None:
    spec = OrderedCategorical(order=LEVELS, basis=Piecewise(breaks=["Mi004"]))
    with pytest.raises(NotImplementedError, match="unpenalized parametric block"):
        resolve_interaction_parent(spec, np.array(LEVELS, dtype=object))


def test_screening_defers_parametric_inner_bases_with_a_reason() -> None:
    from superglm.model.screening_ops import _deferral_reason, _margin_kind

    spec = OrderedCategorical(order=LEVELS, basis=Piecewise(breaks=["Mi004"]))
    assert _margin_kind(spec) is None
    assert "unpenalized parametric block" in _deferral_reason(spec)
    smooth = OrderedCategorical(order=LEVELS, basis=Spline(kind="cr", n_knots=4))
    assert _margin_kind(smooth) == "spline"
