"""Specials: levels held out of the smooth and fitted as free level effects."""

import numpy as np
import pandas as pd
import pytest

from superglm import OrderedCategorical, Spline, SuperGLM

ORDERED = [str(i) for i in range(1, 11)]
SPECIAL = "MISSING"


def _oc(**kwargs):
    params = dict(order=list(ORDERED), specials=[SPECIAL], basis=Spline(kind="ps", k=8))
    params.update(kwargs)
    return OrderedCategorical(**params)


# Today `specials` is not a parameter at all, so construction raises TypeError.
def test_specials_are_held_out_of_the_smooth_levels():
    spec = _oc()
    assert spec._specials == [SPECIAL]
    assert spec._smooth_levels == ORDERED
    assert spec._ordered_levels == ORDERED + [SPECIAL]
    assert SPECIAL not in spec._level_to_value
    assert set(spec._level_to_value) == set(ORDERED)
    assert spec._n_levels == len(ORDERED)
    assert spec.has_specials is True
    # A special is a level of the column, so predict-time validation must accept
    # it rather than reject it as unseen.
    assert SPECIAL in spec._known_levels


def test_no_specials_leaves_everything_unchanged():
    spec = OrderedCategorical(order=list(ORDERED), basis=Spline(kind="ps", k=8))
    assert spec._specials == []
    assert spec._smooth_levels == ORDERED
    assert spec._ordered_levels == ORDERED
    assert spec.has_specials is False


# Each of these raises nothing today — `specials` does not exist, and once it
# does, the naive implementation accepts all of them.
def test_label_in_both_order_and_specials_is_popped_from_order():
    spec = OrderedCategorical(
        order=[SPECIAL] + list(ORDERED), specials=[SPECIAL], basis=Spline(kind="ps", k=8)
    )
    assert spec._smooth_levels == ORDERED
    assert SPECIAL not in spec._level_to_value
    # Positions are computed over the survivors, so band 1 is at 0.0 and band 10 at 1.0.
    assert spec._level_to_value["1"] == pytest.approx(0.0)
    assert spec._level_to_value["10"] == pytest.approx(1.0)


def test_label_in_both_values_and_specials_is_popped_from_values():
    spec = OrderedCategorical(
        values={SPECIAL: -1.0, "a": 1.0, "b": 2.0, "c": 3.0},
        specials=[SPECIAL],
        basis=Spline(kind="ps", k=5),
    )
    assert spec._smooth_levels == ["a", "b", "c"]
    assert SPECIAL not in spec._level_to_value


def test_non_str_special_is_coerced_and_popped_from_order():
    # Level labels are `str` everywhere else in this file, so an int special
    # must not survive as a second, un-popped copy of the same level.
    spec = OrderedCategorical(
        order=["1", "2", "3", "4", "5", "9"], specials=[9], basis=Spline(kind="ps", k=5)
    )
    assert spec._specials == ["9"]
    assert spec._smooth_levels == ["1", "2", "3", "4", "5"]
    assert spec._ordered_levels == ["1", "2", "3", "4", "5", "9"]
    assert "9" not in spec._level_to_value


def test_special_is_popped_from_a_non_str_order_by_string_match():
    # The mirror case: `order=` holds non-str labels. Matching by `str` on both
    # sides is what stops level 9 being smoothed *and* claimed free.
    spec = OrderedCategorical(order=[1, 2, 3, 9], specials=["9"], basis=Spline(kind="ps", k=5))
    assert spec._specials == ["9"]
    assert [str(lev) for lev in spec._smooth_levels] == ["1", "2", "3"]
    assert 9 not in spec._level_to_value
    assert "9" not in spec._level_to_value


def test_special_is_popped_from_non_str_values_keys():
    # Same string match on the `values=` path, where the label is a dict key.
    spec = OrderedCategorical(
        values={1: 1.0, 2: 2.0, 3: 3.0, 9: -1.0}, specials=[9], basis=Spline(kind="ps", k=5)
    )
    assert spec._specials == ["9"]
    assert [str(lev) for lev in spec._smooth_levels] == ["1", "2", "3"]
    assert 9 not in spec._level_to_value
    assert "9" not in spec._level_to_value


def test_duplicate_special_is_rejected():
    with pytest.raises(ValueError, match="Duplicate special level"):
        _oc(specials=[SPECIAL, SPECIAL])


def test_fewer_than_two_smooth_levels_is_rejected():
    with pytest.raises(ValueError, match="at least two"):
        OrderedCategorical(order=["a", SPECIAL], specials=[SPECIAL], basis=Spline(kind="ps", k=5))


def test_specials_with_step_basis_is_rejected():
    with pytest.raises(ValueError, match="basis='step'"):
        OrderedCategorical(order=list(ORDERED), specials=[SPECIAL], basis="step")


def test_explicit_special_base_is_rejected():
    with pytest.raises(ValueError, match="reporting base"):
        _oc(base=SPECIAL)


def test_non_str_base_naming_a_special_is_rejected():
    # The base check runs against the coerced special set, so `base=9` must be
    # caught here rather than surfacing later as "Base '9' not found in levels".
    with pytest.raises(ValueError, match="reporting base"):
        OrderedCategorical(order=[1, 2, 3, 9], specials=[9], base=9, basis=Spline(kind="ps", k=5))


def test_grouping_that_merges_a_special_is_rejected():
    # The spec's validation table forbids mixing a special with ordered levels
    # in one group, but only the editor's collapse path enforces it. Built
    # directly, the special is silently smoothed inside group "6+MISSING"
    # while `_specials` still lists it as free — an inconsistent spec state
    # with no error anywhere.
    from superglm.features.grouping import collapse_levels

    grouping = collapse_levels(
        np.array(ORDERED + [SPECIAL], dtype=object),
        groups={"6+MISSING": ["6", SPECIAL]},
        order=ORDERED + [SPECIAL],
    )
    with pytest.raises(ValueError, match="free level"):
        _oc(grouping=grouping)


def test_grouping_that_renames_a_special_is_rejected():
    # A one-member group is still a rename: 'MISSING' becomes 'UNKNOWN', which
    # joins the grouped smooth levels with no numeric position while `_specials`
    # still names 'MISSING'. `_smooth_levels` would then hold a level absent
    # from `_level_to_value`, and `_n_levels` would count it.
    from superglm.features.grouping import collapse_levels

    grouping = collapse_levels(
        np.array(ORDERED + [SPECIAL], dtype=object),
        groups={"UNKNOWN": [SPECIAL]},
        order=ORDERED + [SPECIAL],
    )
    with pytest.raises(ValueError, match="free level"):
        _oc(grouping=grouping)


def test_grouping_that_collapses_every_ordered_level_is_rejected():
    # The at-least-two-smooth-levels check runs on the grouped level list, so a
    # grouping that leaves one smooth level is refused rather than reaching the
    # spline build with a single distinct position.
    from superglm.features.grouping import collapse_levels

    grouping = collapse_levels(
        np.array(ORDERED + [SPECIAL], dtype=object),
        groups={"all": list(ORDERED)},
        order=ORDERED + [SPECIAL],
    )
    with pytest.raises(ValueError, match="at least two"):
        _oc(grouping=grouping)


# Today `_choose_base` iterates `_ordered_levels`, so once specials are appended
# there, `most_exposed` picks MISSING whenever it dominates exposure — and it
# usually does on a real book.
def test_most_exposed_base_never_selects_a_special():
    spec = _oc()
    x = np.array(["1"] * 10 + [SPECIAL] * 1000, dtype=object)
    weight = np.ones(len(x))
    spec._choose_base(x, weight)
    assert spec._base_level != SPECIAL
    assert spec._base_level in ORDERED
    # The non-base list feeds relativity tables, screening and the editor, so a
    # special must not leak into it either.
    assert SPECIAL not in spec._non_base
    assert spec._non_base == [lev for lev in ORDERED if lev != spec._base_level]


def test_choose_base_reselects_when_a_special_is_already_the_base():
    # The early return must not accept a stale special as the base: a spec
    # cloned from one whose base was set before specials existed would keep it.
    spec = _oc()
    spec._base_level = SPECIAL
    x = np.array(["1"] * 10 + [SPECIAL] * 1000, dtype=object)
    spec._choose_base(x, np.ones(len(x)))
    assert spec._base_level in ORDERED
    assert SPECIAL not in spec._non_base


def _fit_frame(n=4000, seed=11):
    rng = np.random.default_rng(seed)
    band = rng.choice(ORDERED + [SPECIAL], size=n)
    exposure = rng.gamma(shape=4.0, scale=0.25, size=n)
    t = {lv: i / (len(ORDERED) - 1) for i, lv in enumerate(ORDERED)}
    log_rel = np.array([0.6 * (1 - np.exp(-3 * t[b])) if b != SPECIAL else -0.55 for b in band])
    claims = rng.poisson(exposure * np.exp(np.log(0.08) + log_rel))
    return pd.DataFrame({"band": band, "exposure": exposure, "freq": claims / exposure})


# build() has never returned more than one GroupInfo for an OC term.
def test_build_returns_spline_block_then_special_block():
    frame = _fit_frame()
    spec = _oc()
    infos = spec.build(frame["band"].to_numpy(), frame["exposure"].to_numpy())
    assert isinstance(infos, list) and len(infos) == 2
    spline_info, special_info = infos
    assert spline_info.subgroup_name is None
    assert special_info.subgroup_name == "special"
    assert special_info.n_cols == 1
    assert special_info.penalized is False
    assert special_info.penalty_matrix is None
    assert special_info.projection is None
    assert special_info.reparametrize is False


def test_special_indicator_columns_follow_the_declared_special_order():
    # The frozen interface says indicator column j is `spec._specials[j]`. With a
    # single special every column order is the right one, so this pins it on two.
    frame = _fit_frame()
    band = frame["band"].to_numpy()
    rng = np.random.default_rng(3)
    second = "REFUSED"
    band = np.where(
        (band == "10") & (rng.random(len(band)) < 0.5), second, band.astype(object)
    ).astype(object)
    spec = _oc(specials=[SPECIAL, second])
    _, special_info = spec.build(band, frame["exposure"].to_numpy())
    assert spec._specials == [SPECIAL, second]
    assert special_info.n_cols == 2
    indicators = np.asarray(special_info.columns.todense())
    for j, lev in enumerate(spec._specials):
        assert np.array_equal(indicators[:, j] == 1.0, band == lev)


def test_spline_block_is_zero_on_special_rows():
    frame = _fit_frame()
    spec = _oc()
    spline_info, special_info = spec.build(frame["band"].to_numpy(), frame["exposure"].to_numpy())
    is_special = (frame["band"] == SPECIAL).to_numpy()
    spline_cols = np.asarray(spline_info.columns.todense())
    assert np.allclose(spline_cols[is_special], 0.0)
    assert not np.allclose(spline_cols[~is_special], 0.0)
    indicator = np.asarray(special_info.columns.todense()).ravel()
    assert np.array_equal(indicator == 1.0, is_special)


def test_expanded_rows_carry_their_own_rows_basis():
    # Zeroing the special rows is only half of "row-expanded with zeros": each
    # surviving row must hold the basis of *its own* level. A scatter that
    # permutes the ordered rows among their own positions leaves the zero
    # pattern, the column sums and the Gram untouched, so nothing above sees it
    # — but every row would then be fitted against another row's basis.
    frame = _fit_frame()
    band = frame["band"].to_numpy()
    spec = _oc()
    spline_info, _ = spec.build(band, frame["exposure"].to_numpy())
    ordered = band != SPECIAL
    X = np.asarray(spline_info.columns.todense())[ordered]
    labels = band[ordered]

    seen = {}
    for lev in ORDERED:
        rows = X[labels == lev]
        assert len(rows) > 0
        # The basis row is a function of the level alone, so rows sharing a
        # label must be identical — under a permuted scatter they are not.
        np.testing.assert_allclose(rows, np.repeat(rows[:1], len(rows), axis=0), atol=1e-12)
        seen[lev] = rows[0]
    # ...and distinct levels must not share a basis row, or the check is vacuous.
    for a, b in zip(ORDERED, ORDERED[1:]):
        assert not np.allclose(seen[a], seen[b])


def test_declared_special_absent_from_training_data_is_rejected():
    frame = _fit_frame()
    ordered_only = frame[frame["band"] != SPECIAL]
    spec = _oc()
    with pytest.raises(ValueError, match="never observed"):
        spec.build(ordered_only["band"].to_numpy(), ordered_only["exposure"].to_numpy())


# The construction is only legitimate if [1 | centered spline | indicators] is
# full rank. A centered basis cannot reproduce a constant, so no indicator is
# recoverable from the other columns — this pins that argument.
#
# ``columns`` is the RAW B-spline basis; the centering lives in ``projection``,
# and the DM builder only ever forms ``columns @ projection @ R_inv_local``
# (dm_builder._process_info). Rank must therefore be asserted on ``columns @
# projection``: the raw basis is a partition of unity, so on the raw block
# ``1 == rowsum(B) + indicator`` for reasons that predate specials entirely.
#
# This is a NECESSARY condition only, not a guard on the ordered-row build: the
# constant is generically outside a centered block's span however that block was
# centered, so full rank also holds for the rejected build (spline over all rows
# with a fabricated coordinate for the specials, zeroed afterwards). The
# discriminating property is the one below —
# ``test_identifiability_constraint_holds_on_the_ordered_rows``, which the
# rejected build fails outright.
def test_assembled_design_with_intercept_is_full_rank():
    frame = _fit_frame()
    spec = _oc()
    spline_info, special_info = spec.build(frame["band"].to_numpy(), frame["exposure"].to_numpy())
    assert spline_info.projection is not None
    centered = np.asarray(spline_info.columns.todense()) @ spline_info.projection
    assert centered.shape[1] == spline_info.n_cols
    design = np.column_stack(
        [
            np.ones(len(frame)),
            centered,
            np.asarray(special_info.columns.todense()),
        ]
    )
    assert np.linalg.matrix_rank(design) == design.shape[1]


def test_identifiability_constraint_holds_on_the_ordered_rows():
    # The reason the spline is built on the ordered rows rather than on all rows
    # and zeroed afterwards: build_identifiability_projection forms its constraint
    # as a column sum over the rows it was handed. Built on all rows, this sum
    # would be zero over ALL rows and non-zero over the ordered ones once the
    # special rows are zeroed, so the centered block would still carry a constant.
    frame = _fit_frame()
    spec = _oc()
    spline_info, _ = spec.build(frame["band"].to_numpy(), frame["exposure"].to_numpy())
    ordered = (frame["band"] != SPECIAL).to_numpy()
    centered = np.asarray(spline_info.columns.todense()) @ spline_info.projection
    np.testing.assert_allclose(centered[ordered].sum(axis=0), 0.0, atol=1e-8)


def test_ssp_gram_of_the_zero_filled_block_is_on_the_all_row_scale():
    # The SSP contract (compute_R_inv's docstring) is X'WX / sum(sample_weight)
    # ~ I over EVERY row, and it is what puts each block of the assembled normal
    # equations on one common scale. A zero-filled block satisfies it without
    # help: its zero rows contribute nothing to X'WX but do contribute to the
    # normaliser, exactly as for any block whose basis is small on some rows.
    # Normalising this block by its own rows' weight sum instead would rescale
    # it by the special's exposure share (~6x here) against every other column,
    # so both halves below are asserted: the contract, and scale parity with an
    # ordinary full-support block fitted on the same weights.
    from superglm.dm_builder import _process_info

    frame = _fit_frame(n=6000, seed=17)
    band = frame["band"].to_numpy()
    is_special = band == SPECIAL
    weight = np.where(is_special, 50.0, 1.0)  # the special dominates exposure

    spec = _oc()
    spline_info, _ = spec.build(band, weight)

    gm, _, _ = _process_info(spline_info, sample_weight=weight, lambda2=0.0)
    X = np.asarray(gm.toarray(), dtype=np.float64)
    assert np.allclose(X[is_special], 0.0)

    gram = X.T @ (weight[:, None] * X)
    # atol is loose against the 1e-8 ridge's imprint (it lifts the identity by
    # 1e-8 / smallest Gram eigenvalue) and tight against a normalisation by the
    # ordered-row weight sum, which is a pure scale error of ~6x throughout.
    np.testing.assert_allclose(gram / weight.sum(), np.eye(gram.shape[0]), atol=1e-3)

    full_support = Spline(kind="ps", k=8).build(np.linspace(0.0, 1.0, len(band)))
    gm_ref, _, _ = _process_info(full_support, sample_weight=weight, lambda2=0.0)
    X_ref = np.asarray(gm_ref.toarray(), dtype=np.float64)
    gram_ref = X_ref.T @ (weight[:, None] * X_ref)
    np.testing.assert_allclose(np.diag(gram), np.diag(gram_ref), rtol=1e-3)


# transform() has always returned only the spline's columns, so its width is
# n_spline_cols today and the assertions below are off by len(specials).
def test_transform_emits_spline_then_special_columns():
    frame = _fit_frame()
    spec = _oc()
    spline_info, special_info = spec.build(frame["band"].to_numpy(), frame["exposure"].to_numpy())
    probe = np.array(ORDERED + [SPECIAL], dtype=object)
    out = spec.transform(probe)
    # transform() emits the basis at the inner spline's current width, which is
    # the built block's column count until the identifiability reparametrisation
    # is pushed back in; n_cols is already post-projection, so it is the wrong
    # yardstick here.
    n_spline = spline_info.columns.shape[1]
    assert out.shape == (len(probe), n_spline + special_info.n_cols)
    # Special rows are zero across the spline block, ordered rows zero across the indicators.
    assert np.allclose(out[-1, :n_spline], 0.0)
    assert np.allclose(out[:-1, n_spline:], 0.0)
    assert out[-1, n_spline] == 1.0
    assert not np.allclose(out[:-1, :n_spline], 0.0)


def test_split_beta_partitions_by_block_width():
    frame = _fit_frame()
    spec = _oc()
    spline_info, special_info = spec.build(frame["band"].to_numpy(), frame["exposure"].to_numpy())
    # The full width the term actually consumes: the inner spline has not had
    # the identifiability reparametrisation pushed back in, so its block is as
    # wide as it was built, and n_cols (post-projection) is the wrong yardstick.
    n_spline = spline_info.columns.shape[1]
    beta = np.arange(n_spline + special_info.n_cols, dtype=np.float64)
    spline_beta, special_beta = spec._split_beta(beta)
    # Assert the block BOUNDARY, not the lengths: _split_beta strips the trailing
    # n_special from whatever it is handed, so a length check is a tautology.
    np.testing.assert_array_equal(spline_beta, beta[: -special_info.n_cols])
    np.testing.assert_array_equal(special_beta, beta[-special_info.n_cols :])
    assert special_beta[0] == beta[-1]


def test_split_beta_rejects_a_spline_only_vector():
    # The realistic bad input a width guard exists to catch: a caller hands over
    # the spline block alone. It is longer than n_special, so a one-sided
    # `len(beta) < n_special` check passes it through and the last spline
    # coefficients are silently reinterpreted as free level effects.
    frame = _fit_frame()
    spec = _oc()
    spline_info, _ = spec.build(frame["band"].to_numpy(), frame["exposure"].to_numpy())
    spline_only = np.arange(spline_info.columns.shape[1], dtype=np.float64)
    with pytest.raises(ValueError, match="coefficients but its blocks"):
        spec._split_beta(spline_only)


def test_a_non_string_special_label_builds_and_transforms():
    # False today: `specials=[9]` is str-coerced to "9" at construction but
    # `_special_mask` compares the raw column against "9", so every row of an
    # integer-labelled column misses the indicator; the special rows then land
    # in the spline's ordered set and _map_to_numeric emits NaN.
    spec = OrderedCategorical(order=[1, 2, 3, 4, 5, 9], specials=[9], basis=Spline(kind="ps", k=5))
    x = np.array([1, 2, 3, 4, 5, 9, 9], dtype=object)
    spline_info, special_info = spec.build(x, np.ones(len(x)))
    assert special_info.n_cols == 1
    indicator = np.asarray(special_info.columns.todense()).ravel()
    np.testing.assert_array_equal(indicator == 1.0, np.array(x) == 9)
    out = spec.transform(x)
    n_spline = spline_info.columns.shape[1]
    assert out.shape == (len(x), n_spline + 1)
    np.testing.assert_array_equal(out[:, -1] == 1.0, np.array(x) == 9)
    assert np.allclose(out[-2:, :n_spline], 0.0)


def test_a_non_string_special_label_builds_on_a_float_column():
    # The mirror case, and the half a str-only mask misses: on a FLOAT column
    # `pd.Series(9.0).astype(str)` renders "9.0", which never equals the coerced
    # "9". The special rows then miss the indicator, the build sees an all-zero
    # column and raises "never observed" on data that plainly contains them.
    # Plain OC takes a float column in its stride (`_level_to_value[9.0]` hits
    # the int key), so the specials path must not be narrower.
    spec = OrderedCategorical(order=[1, 2, 3, 4, 5, 9], specials=[9], basis=Spline(kind="ps", k=5))
    x = np.array([1, 2, 3, 4, 5, 9, 9], dtype=float)
    spline_info, special_info = spec.build(x, np.ones(len(x)))
    assert special_info.n_cols == 1
    indicator = np.asarray(special_info.columns.todense()).ravel()
    np.testing.assert_array_equal(indicator == 1.0, x == 9.0)
    out = spec.transform(x)
    n_spline = spline_info.columns.shape[1]
    assert out.shape == (len(x), n_spline + 1)
    np.testing.assert_array_equal(out[:, -1] == 1.0, x == 9.0)
    assert np.allclose(out[-2:, :n_spline], 0.0)


def test_numeric_order_with_a_string_special_builds_and_transforms():
    # The commonest real shape of all -- numeric bands beside a labelled special,
    # `order=[1..6], specials=["MISSING"]` -- and it does not fit today. The
    # ungrouped path validates raw column labels, and `_validate_categorical_levels`
    # calls `np.unique`, which SORTS: on an object column holding both ints and
    # "MISSING" that raises `TypeError: '<' not supported between 'int' and 'str'`
    # before `_special_mask` ever gets to hold the special out. Nothing about the
    # domain is actually wrong -- only the validator's insistence on ordering it.
    spec = OrderedCategorical(
        order=[1, 2, 3, 4, 5, 6], specials=["MISSING"], basis=Spline(kind="ps", k=5)
    )
    x = np.array([1, 2, 3, 4, 5, 6, "MISSING", "MISSING"], dtype=object)
    spline_info, special_info = spec.build(x, np.ones(len(x)))
    assert special_info.n_cols == 1
    indicator = np.asarray(special_info.columns.todense()).ravel()
    np.testing.assert_array_equal(indicator == 1.0, np.array(x) == "MISSING")
    out = spec.transform(x)
    n_spline = spline_info.columns.shape[1]
    assert out.shape == (len(x), n_spline + 1)
    # The special rows carry the indicator and a zeroed spline block; the ordered
    # rows carry a real basis, so this cannot pass on an all-zero design.
    np.testing.assert_array_equal(out[:, -1] == 1.0, np.array(x) == "MISSING")
    assert np.allclose(out[-2:, :n_spline], 0.0)
    assert not np.allclose(out[:-2, :n_spline], 0.0)


def test_an_unseen_level_still_reports_against_a_mixed_type_domain():
    # The other half: the validator must still REJECT genuinely unseen levels when
    # the domain is mixed, and must be able to render the message. Formatting it
    # sorts both the unseen set and the known domain, which is the same TypeError
    # one layer down -- so a fix that only stops np.unique sorting turns a clean
    # ValueError into a TypeError from the error path itself.
    spec = OrderedCategorical(
        order=[1, 2, 3, 4, 5, 6], specials=["MISSING"], basis=Spline(kind="ps", k=5)
    )
    x = np.array([1, 2, 3, 4, 5, 6, "MISSING"], dtype=object)
    spec.build(x, np.ones(len(x)))
    with pytest.raises(ValueError, match="unseen categorical levels"):
        spec.transform(np.array([1, 2, "NOT_A_LEVEL"], dtype=object))


# score() and reconstruct() forward the whole vector to the inner spline today,
# so with specials present they read special coefficients as spline ones.
def test_score_uses_the_free_coefficient_on_special_rows():
    frame = _fit_frame()
    spec = _oc()
    spline_info, special_info = spec.build(frame["band"].to_numpy(), frame["exposure"].to_numpy())
    # The spec has not had the identifiability reparametrisation pushed back in,
    # so the inner spline still expects one coefficient per built column.
    n_spline = spline_info.columns.shape[1]
    # A nonzero, non-constant spline block. With an all-zero one an ordered row
    # reads 0.0 whether the smooth is evaluated or dropped from score() outright,
    # so the whole ordered branch could be deleted with this test still green.
    beta = np.linspace(0.1, 0.9, n_spline + special_info.n_cols)
    beta[-1] = -0.55
    scored = spec.score(np.array(["1", SPECIAL], dtype=object), beta)
    assert scored[1] == pytest.approx(-0.55)
    # The ordered row carries the inner smooth evaluated at its own level...
    expected = float(spec._spline.score(np.array([spec._level_to_value["1"]]), beta[:n_spline])[0])
    assert scored[0] == pytest.approx(expected)
    assert abs(scored[0]) > 1e-6  # ...and the smooth is genuinely nonzero there
    # ...and none of the special's coefficient: moving it must not move the
    # ordered row. An all-zero-except-the-last beta cannot see that.
    moved = beta.copy()
    moved[-1] = 4.0
    assert spec.score(np.array(["1"], dtype=object), moved)[0] == pytest.approx(expected)


def test_reconstruct_reports_every_level_and_flags_the_specials():
    frame = _fit_frame()
    spec = _oc()
    spline_info, special_info = spec.build(frame["band"].to_numpy(), frame["exposure"].to_numpy())
    n_spline = spline_info.columns.shape[1]
    # A nonzero spline block, so the base shift is live: with an all-zero one
    # `base_shift` is exactly 0.0 and the spec's `beta_special - f(base)`
    # reporting rule can be dropped from _reconstruct_spline unnoticed.
    beta = np.zeros(n_spline + special_info.n_cols)
    beta[:n_spline] = 0.3
    beta[-1] = -0.55
    raw = spec.reconstruct(beta)
    assert raw["levels"] == ORDERED + [SPECIAL]
    assert raw["special_levels"] == [SPECIAL]
    assert set(raw["level_relativities"]) == set(ORDERED + [SPECIAL])
    base_shift = spec._base_log_effect(beta)
    assert base_shift != pytest.approx(0.0)
    # The spec's reporting rule: a special reports beta_special - f(base), on the
    # same scale as every smooth level, which is anchored at zero on the base.
    assert raw["level_log_relativities"][SPECIAL] == pytest.approx(-0.55 - base_shift)
    assert raw["level_relativities"][SPECIAL] == pytest.approx(np.exp(-0.55 - base_shift))
    assert raw["level_log_relativities"][spec._base_level] == pytest.approx(0.0)
    # A special never receives a coordinate on the smooth's axis.
    assert SPECIAL not in raw["level_values"]


# Nothing enforces this today because the feature does not exist. It is the
# claim the design rests on: a free level effect makes those rows uninformative
# for the term's OWN coefficients, so the fitted curve must not move.
#
# Scope, because the unqualified version of that sentence is not true: it holds
# exactly for this fixture -- one predictor, one smoothing parameter -- and is
# only approximate once another correlated predictor is present, because the
# special's rows still reach the other coefficients through the shared IRLS
# weights (measured at ~1e-3 with one imbalanced factor at a 5% special share),
# and again once REML re-selects lambda. The fixture below is deliberately
# single-predictor so the strong claim is the one under test.
def test_adding_a_special_does_not_move_the_fitted_curve():
    frame = _fit_frame(n=8000, seed=3)
    ordered_only = frame[frame["band"] != SPECIAL].reset_index(drop=True)

    without = SuperGLM(
        family="poisson",
        link="log",
        features={"band": OrderedCategorical(order=list(ORDERED), basis=Spline(kind="ps", k=8))},
    )
    without.fit_reml(
        ordered_only[["band"]],
        ordered_only["freq"].to_numpy(),
        sample_weight=ordered_only["exposure"].to_numpy(),
    )

    with_special = SuperGLM(family="poisson", link="log", features={"band": _oc()})
    with_special.fit_reml(
        frame[["band"]], frame["freq"].to_numpy(), sample_weight=frame["exposure"].to_numpy()
    )

    a = without.term_inference("band", with_se=False)
    b = with_special.term_inference("band", with_se=False)
    rel_a = dict(zip([str(v) for v in a.levels], np.asarray(a.relativity, dtype=float)))
    rel_b = dict(zip([str(v) for v in b.levels], np.asarray(b.relativity, dtype=float)))
    for lev in ORDERED:
        assert rel_b[lev] == pytest.approx(rel_a[lev], rel=2e-2)


def test_a_special_carrying_no_weight_is_refused_like_an_absent_one():
    # False today: the presence check reads the raw boolean mask, so a special
    # that appears only on zero-weight rows counts as observed. In the weighted
    # fit its indicator contributes nothing to X'WX, so the unpenalized
    # coefficient is exactly as unidentifiable as the absent case the branch
    # immediately above already rejects -- the model just does not say so.
    spec = OrderedCategorical(
        order=["1", "2", "3", "4", "5"], specials=["MISSING"], basis=Spline(kind="ps", k=5)
    )
    x = np.array(["1", "2", "3", "4", "5", "MISSING", "MISSING"], dtype=object)
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="carry no weight"):
        spec.build(x, weights)


def test_a_special_with_some_positive_weight_still_builds():
    # The other side of the gate: one weighted row is enough to identify the
    # coefficient, so the check must not reject a merely-thin special.
    spec = OrderedCategorical(
        order=["1", "2", "3", "4", "5"], specials=["MISSING"], basis=Spline(kind="ps", k=5)
    )
    x = np.array(["1", "2", "3", "4", "5", "MISSING", "MISSING"], dtype=object)
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.25])
    _, special_info = spec.build(x, weights)
    assert special_info.n_cols == 1


def test_a_grouping_that_omits_a_special_is_refused():
    # `_require_no_grouped_specials` refuses a grouping that MERGES or RENAMES a
    # special, but not one that simply fails to COVER it. Construction survives
    # because `_known_levels` re-adds the special, and then build()'s
    # `pd.Series(x).map(original_to_group)` yields NaN for those rows: the
    # special mask misses them and they reach `_map_to_numeric` as NaN, dying
    # inside scipy with "Array must not contain infs or nans" -- a message that
    # says nothing about the grouping that caused it.
    from superglm.features.grouping import collapse_levels

    grouping = collapse_levels(
        np.array(ORDERED, dtype=object),
        groups={"1+2": ["1", "2"]},
        order=list(ORDERED),
    )
    with pytest.raises(ValueError, match="does not cover"):
        _oc(grouping=grouping)


def test_a_numerically_equal_special_is_popped_from_a_float_order():
    # False today. Popping a special out of `order=` compares `str(level)` against
    # the coerced special set, so `order=[1.0, 2.0, 9.0]` with `specials=[9]`
    # keeps 9.0 -- "9.0" != "9" -- while `_special_mask` DOES match raw 9.0 rows
    # numerically. The level is then both smoothed and free: the spline is built
    # with a phantom position that no row ever occupies, and the same level is
    # reported twice, once as an unobserved smooth level and once as the free one.
    #
    # The str view and the raw view must agree on which levels are special; this
    # is the one place they disagree.
    spec = OrderedCategorical(
        order=[1.0, 2.0, 3.0, 4.0, 9.0], specials=[9], basis=Spline(kind="ps", k=5)
    )
    assert spec._specials == ["9"]
    assert [str(lev) for lev in spec._smooth_levels] == ["1.0", "2.0", "3.0", "4.0"]
    assert 9.0 not in spec._level_to_value
    assert "9.0" not in spec._level_to_value
    assert spec._n_levels == 4

    # And it must still build: every 9.0 row belongs to the free block, and the
    # smooth carries the other four.
    x = np.array([1.0, 2.0, 3.0, 4.0, 9.0, 9.0], dtype=float)
    _, special_info = spec.build(x, np.ones(len(x)))
    indicator = np.asarray(special_info.columns.todense()).ravel()
    np.testing.assert_array_equal(indicator == 1.0, x == 9.0)


def test_a_raw_matched_special_reports_the_label_its_domain_uses():
    # False today. With `order=[1.0, 2.0, 9.0]` and `specials=[9]`, the fit
    # correctly claims the 9.0 rows as the free level -- but `reconstruct()`
    # reports that level as the coerced string "9" while every smooth level is
    # reported with its raw domain label (1.0, 2.0, ...). Rating tables, editor
    # weights and plot exposure bars aggregate row weights by the column's own
    # labels and then look them up by these reported levels, so the special comes
    # back with zero weight and zero exposure -- silently wrong about the one
    # level the feature exists to fit.
    #
    # The reported label must follow the same convention as its neighbours.
    spec = OrderedCategorical(
        order=[1.0, 2.0, 3.0, 4.0, 9.0], specials=[9], basis=Spline(kind="ps", k=5)
    )
    x = np.array([1.0, 2.0, 3.0, 4.0, 9.0, 9.0], dtype=float)
    spline_info, special_info = spec.build(x, np.ones(len(x)))
    n = spline_info.columns.shape[1] + special_info.n_cols
    raw = spec.reconstruct(np.zeros(n, dtype=np.float64))

    assert raw["special_levels"] == [9.0]
    assert raw["levels"][-1] == 9.0
    assert 9.0 in raw["level_relativities"]
    # And every level in the report is a label the raw column actually contains.
    assert set(raw["levels"]) <= set(x.tolist())
