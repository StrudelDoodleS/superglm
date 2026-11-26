"""Neighbourhoods for ``fit_ncv``: constructors against enumerated splitters, every
validation message, and the integer check for factor levels a neighbourhood removes whole."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse
from sklearn.model_selection import (
    KFold,
    LeaveOneGroupOut,
    LeaveOneOut,
    PredefinedSplit,
    RepeatedKFold,
    TimeSeriesSplit,
)
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

from superglm import SuperGLM
from superglm._group_matrix._group_matrix_core import CategoricalGroupMatrix, DenseGroupMatrix
from superglm.features.categorical import Categorical
from superglm.features.numeric import Numeric
from superglm.features.spline import Spline
from superglm.model.reml_setup import collect_reml_groups
from superglm.ncv import (
    FEW_NEIGHBOURHOODS,
    LevelConflict,
    build_neighbourhoods,
    from_groups,
    from_neighbours,
    from_splitter,
    single_rows,
    unidentified_levels,
)
from superglm.reml.penalty_algebra import build_penalty_context


def enumerated(nb):
    """(removed rows, scored rows) per neighbourhood, read back through the accessors."""
    out = []
    for k in range(nb.m):
        drop = nb.drop(k)
        pos = nb.scored_positions(k)
        out.append((drop.tolist(), drop.tolist() if pos is None else drop[pos].tolist()))
    return out


def splitter_reference(cv, X, groups=None):
    """(complement of train, sorted test) for each split, enumerated row by row."""
    n = len(X)
    return [
        (sorted(set(range(n)) - set(train.tolist())), sorted(test.tolist()))
        for train, test in cv.split(X, None, groups)
    ]


class Splits:
    """A splitter that yields fixed ``(train, test)`` pairs and counts its ``split`` calls."""

    def __init__(self, *pairs):
        self.pairs = pairs
        self.calls = 0

    def split(self, X, y=None, groups=None):
        self.calls += 1
        yield from ((np.asarray(a), np.asarray(b)) for a, b in self.pairs)


def test_single_rows_equal_leave_one_out():
    X = np.zeros((7, 1))
    nb = single_rows(7)
    assert nb.drop_ptr is None and nb.drop_rows is None and nb.score_ptr is None
    assert nb.m == 7
    assert enumerated(nb) == splitter_reference(LeaveOneOut(), X)
    assert nb.sizes().tolist() == [1] * 7
    assert (nb.rows_scored, nb.rows_scored_more_than_once) == (7, 0)


def test_groups_equal_leave_one_group_out():
    rng = np.random.default_rng(0)
    groups = rng.choice(["b", "a", "d", "c", "e"], 200)
    X = np.zeros((200, 1))
    nb = from_groups(groups, 200)
    through_splitter = from_splitter(LeaveOneGroupOut(), X, None, groups)
    assert enumerated(nb) == splitter_reference(LeaveOneGroupOut(), X, groups)
    np.testing.assert_array_equal(nb.drop_ptr, through_splitter.drop_ptr)
    np.testing.assert_array_equal(nb.drop_rows, through_splitter.drop_rows)
    assert nb.score_ptr is None and through_splitter.score_ptr is None
    assert nb.labels.tolist() == ["a", "b", "c", "d", "e"]
    assert nb.drop_rows.dtype == np.intp and nb.drop_ptr.dtype == np.intp
    assert not nb.drop_rows.flags.writeable
    assert nb.describe(2) == "group 'c'"


def test_kfold_scores_all_it_removes():
    X = np.zeros((23, 1))
    cv = KFold(4, shuffle=True, random_state=3)
    nb = from_splitter(cv, X, None, None)
    assert nb.form == "cv" and nb.source == "KFold" and nb.labels.tolist() == [0, 1, 2, 3]
    assert nb.score_ptr is None
    assert enumerated(nb) == splitter_reference(cv, X)
    assert (nb.rows_scored, nb.rows_scored_more_than_once) == (23, 0)


def test_forward_splitter_scores_strictly_inside_removed():
    X = np.zeros((30, 1))
    cv = TimeSeriesSplit(4)
    nb = from_splitter(cv, X, None, None)
    assert nb.score_ptr is not None and nb.score_pos.dtype == np.intp
    assert enumerated(nb) == splitter_reference(cv, X)
    assert all(len(scored) < len(removed) for removed, scored in enumerated(nb)[:-1])
    first_block = 30 - sum(len(test) for _, test in cv.split(X))
    assert nb.rows_scored == 30 - first_block
    assert nb.rows_scored_more_than_once == 0


def test_repeated_folds_score_rows_more_than_once():
    nb = from_splitter(
        RepeatedKFold(n_splits=3, n_repeats=2, random_state=0), np.zeros((12, 1)), None, None
    )
    assert (nb.m, nb.rows_scored, nb.rows_scored_more_than_once) == (6, 12, 12)


def test_splitter_is_called_once():
    cv = Splits(([2, 3], [0, 1]), ([0, 1], [2, 3]))
    from_splitter(cv, np.zeros((4, 1)), None, None)
    assert cv.calls == 1


def test_neighbours_list_adds_the_row_and_scores_it():
    neighbours = [[1, 1, 2], [], [0, 2], np.array([0, 1], dtype=np.int32), [3]]
    nb = from_neighbours(neighbours, 5)
    expected = [(sorted(set(near) | {i}), [i]) for i, near in enumerate(map(list, neighbours))]
    assert enumerated(nb) == expected
    assert nb.drop_rows.dtype == np.intp and nb.score_pos.dtype == np.intp
    assert nb.score_pos.tolist() == [0, 0, 1, 2, 1]
    assert (nb.rows_scored, nb.rows_scored_more_than_once) == (5, 0)


def test_sparse_neighbours_equal_the_list_form():
    points = np.random.default_rng(1).uniform(size=(60, 2))
    graph = kneighbors_graph(points, 4, mode="distance", include_self=False)
    near = NearestNeighbors(n_neighbors=5).fit(points).kneighbors(points, return_distance=False)
    listed = from_neighbours([row[row != i] for i, row in enumerate(near)], 60)
    for matrix in (graph, scipy.sparse.coo_array(graph), scipy.sparse.csc_matrix(graph)):
        sparse = from_neighbours(matrix, 60)
        np.testing.assert_array_equal(sparse.drop_ptr, listed.drop_ptr)
        np.testing.assert_array_equal(sparse.drop_rows, listed.drop_rows)
        np.testing.assert_array_equal(sparse.score_pos, listed.score_pos)


# ── validation, one message each ──────────────────────────────────────────────

N = 6
ACTIVE = np.ones(N, dtype=bool)
FRAME = np.zeros((N, 1))


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        (
            {"cv": KFold(2), "neighbours": [[]] * N},
            ValueError,
            "takes `cv` or `neighbours`, not both",
        ),
        (
            {"groups": np.arange(N), "neighbours": [[]] * N},
            ValueError,
            "cannot be combined with `neighbours`",
        ),
        (
            {"groups": np.arange(N - 1)},
            ValueError,
            r"one label per row: it has shape \(5,\) for 6 rows",
        ),
        (
            {"groups": [1, 2, None, 1, 2, np.nan]},
            ValueError,
            "2 missing labels, the first at row 2",
        ),
        ({"cv": KFold(2), "groups": np.arange(N + 1)}, ValueError, "one label per row"),
        ({"neighbours": [[]] * (N - 1)}, ValueError, "has 5 entries for 6 rows"),
        ({"neighbours": scipy.sparse.eye(N, N + 1)}, ValueError, "is a 6 x 7 matrix for 6 rows"),
        (
            {"neighbours": [[1], [], [7], [], [], []]},
            ValueError,
            r"`neighbours\[2\]` holds 7, which is not a row",
        ),
        ({"neighbours": [[1], [-1], [], [], [], []]}, ValueError, r"`neighbours\[1\]` holds -1"),
        (
            {"neighbours": [[1], [], [], [0.5], [], []]},
            ValueError,
            r"`neighbours\[3\]` holds values of type float64",
        ),
        ({"cv": 5}, TypeError, "must be a splitter with a `split` method"),
        (
            {"cv": Splits(([0, 1, 2], [3, 4, 9]), ([3, 4, 5], [0, 1, 2]))},
            ValueError,
            r"Split 0 of `cv` \(Splits\) gives the test row 9, which is not a row position from 0 to 5",
        ),
        (
            {"cv": Splits(([0.0, 1.0], [3, 4]), ([3, 4], [0, 1]))},
            ValueError,
            "gives training rows of type float64",
        ),
        ({"groups": ["a"] * N}, ValueError, r"one label only \('a'\)"),
        (
            {"cv": PredefinedSplit([0, 0, 0, -1, -1, -1])},
            ValueError,
            r"`cv` \(PredefinedSplit\) gave 1 splits and fit_ncv needs at least two",
        ),
        (
            {"cv": Splits(([0, 1, 2], [3, 4, 5]), ([0, 1, 2, 3], [3, 4, 5]))},
            ValueError,
            r"Split 1 of `cv` \(Splits\) scores 1 rows it also trains on",
        ),
        (
            {"cv": Splits(([0, 1, 2], [3, 4, 5]), ([0, 1, 2], []))},
            ValueError,
            "Split 1 .* has no test rows",
        ),
    ],
)
def test_refusals(kwargs, error, message):
    with pytest.raises(error, match=message):
        build_neighbourhoods(N, ACTIVE, frame=FRAME, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "where"),
    [
        ({"groups": ["a", "a", "a", "b", "b", "b"]}, "Group 'b'"),
        ({}, "Row 4"),
        ({"neighbours": [[], [], [], [], [3, 5], []]}, "The neighbourhood of row 4"),
        ({"cv": Splits(([0, 1, 2, 3], [4, 5]), ([4, 5], [0, 1, 2, 3]))}, "Split 0 of Splits"),
    ],
)
def test_neighbourhood_removing_every_weighted_row_is_refused(kwargs, where):
    active = np.array([False, False, False, False, True, False])
    if "groups" in kwargs or "neighbours" in kwargs:
        active[5] = True
    with pytest.raises(ValueError, match=f"^{where} removes every row with positive weight"):
        build_neighbourhoods(N, active, frame=FRAME, **kwargs)


@pytest.mark.filterwarnings("error")
def test_few_neighbourhoods_warn_below_ten():
    n = 40
    build_neighbourhoods(n, np.ones(n, dtype=bool), groups=np.arange(n) % FEW_NEIGHBOURHOODS)
    with pytest.warns(
        UserWarning, match="fit_ncv has 9 neighbourhoods; with fewer than 10 the held-out"
    ):
        build_neighbourhoods(
            n, np.ones(n, dtype=bool), groups=np.arange(n) % (FEW_NEIGHBOURHOODS - 1)
        )


# ── the structural check ──────────────────────────────────────────────────────


def brute_force_conflicts(nb, codes, n_levels, active):
    """(level, first neighbourhood) for every level some neighbourhood removes whole, row by row."""
    found = {}
    for k in range(nb.m):
        kept = np.ones(len(codes), dtype=bool)
        kept[nb.drop(k)] = False
        for level in range(n_levels + 1):
            had = bool(np.any(active & (codes == level)))
            if had and not np.any(active & kept & (codes == level)) and level not in found:
                found[level] = k
    return sorted(found.items())


def one_factor_design(codes, n_levels, *, penalised=()):
    dense = np.random.default_rng(0).normal(size=(len(codes), 2))
    dm = SimpleNamespace(
        group_matrices=[DenseGroupMatrix(dense), CategoricalGroupMatrix(codes, n_levels)]
    )
    groups = [SimpleNamespace(name="z"), SimpleNamespace(name="f")]
    return dm, groups, set(penalised)


@pytest.fixture
def factor_rows():
    """400 rows, a four-level factor coded 0..2 with the reference level at sink code 3.

    Level 2 has all its weighted rows in group 7 and one zero-weight row in group 3; level 1
    has three quarters of its rows in group 5, so a share rule would flag it.
    """
    rng = np.random.default_rng(21)
    n = 400
    codes = rng.choice(4, n, p=[0.4, 0.3, 0.2, 0.1])
    groups = rng.integers(0, 20, n)
    active = rng.uniform(size=n) > 0.1
    groups[codes == 2] = 7
    stray = np.flatnonzero(codes == 2)[0]
    groups[stray], active[stray] = 3, False
    level_one = np.flatnonzero((codes == 1) & active)
    groups[level_one[: 3 * len(level_one) // 4]] = 5
    groups[level_one[3 * len(level_one) // 4 :]] = 11
    return codes, groups, active


def test_structural_check_level_in_one_group(factor_rows):
    codes, groups, active = factor_rows
    dm, terms, penalised = one_factor_design(codes, 3)
    nb = from_groups(groups, len(codes))
    found = unidentified_levels(nb, dm, terms, active, penalised)
    assert found == [
        LevelConflict(group_index=1, term="f", level=2, reference=False, neighbourhood=7)
    ]
    assert [(c.level, c.neighbourhood) for c in found] == brute_force_conflicts(
        nb, codes, 3, active
    )
    assert found[0].message(nb, "d") == (
        "Every weighted row of level 'd' of 'f' lies in group 7, so with it held out nothing in the "
        "fit estimates that level; merge the level with another or choose neighbourhoods that "
        "divide its rows."
    )


def test_structural_check_zero_weight_rows_are_not_support(factor_rows):
    codes, groups, active = factor_rows
    dm, terms, penalised = one_factor_design(codes, 3)
    nb = from_groups(groups, len(codes))
    assert unidentified_levels(nb, dm, terms, np.ones(len(codes), dtype=bool), penalised) == []
    assert [c.level for c in unidentified_levels(nb, dm, terms, active, penalised)] == [2]


def test_structural_check_reference_level(factor_rows):
    codes, _, active = factor_rows
    groups = np.random.default_rng(4).integers(0, 20, len(codes))
    groups[codes == 3] = 3
    dm, terms, penalised = one_factor_design(codes, 3)
    nb = from_groups(groups, len(codes))
    found = unidentified_levels(nb, dm, terms, active, penalised)
    assert found == [LevelConflict(1, "f", 3, True, 3)]
    assert [(c.level, c.neighbourhood) for c in found] == brute_force_conflicts(
        nb, codes, 3, active
    )
    assert "the other levels of 'f' add up to the intercept" in found[0].message(nb, "base")


def test_structural_check_single_row_level():
    rng = np.random.default_rng(5)
    codes = rng.choice(3, 120)
    codes[[17, 40]] = 3
    active = np.ones(120, dtype=bool)
    active[40] = False
    dm, terms, penalised = one_factor_design(codes, 3)
    nb = single_rows(120)
    found = unidentified_levels(nb, dm, terms, active, penalised)
    assert found == [LevelConflict(1, "f", 3, True, 17)]
    assert found == unidentified_levels(
        from_groups(np.arange(120), 120), dm, terms, active, penalised
    )
    assert [(c.level, c.neighbourhood) for c in found] == brute_force_conflicts(
        nb, codes, 3, active
    )


def test_structural_check_skips_penalised_and_other_blocks(factor_rows):
    codes, groups, active = factor_rows
    dm, terms, _ = one_factor_design(codes, 3)
    nb = from_groups(groups, len(codes))
    assert unidentified_levels(nb, dm, terms, active, {1}) == []


def test_structural_check_on_a_fitted_design_is_where_the_leave_out_design_loses_rank():
    """On a built model, the conflicts are exactly the neighbourhoods whose kept weighted rows
    leave the intercept and the factor's columns rank deficient."""
    rng = np.random.default_rng(21)
    n = 300
    df = pd.DataFrame(
        {
            "x1": rng.uniform(0, 1, n),
            "z": rng.normal(size=n),
            "f": rng.choice(list("abcd"), n, p=[0.4, 0.3, 0.2, 0.1]),
        }
    )
    y = np.sin(5 * df.x1.to_numpy()) + 0.2 * df.z.to_numpy() + rng.normal(0, 0.3, n)
    model = SuperGLM(
        features={"x1": Spline(kind="ps", n_knots=6), "z": Numeric(), "f": Categorical()},
        family="gaussian",
        selection_penalty=0,
        spline_penalty=1.0,
    )
    model.fit(df, y)
    dm = model._dm
    penalties, _, _ = build_penalty_context(
        dm.group_matrices, collect_reml_groups(model._groups, dm.group_matrices)
    )
    penalised = {pc.group_index for pc in penalties}
    (g,) = [i for i, gm in enumerate(dm.group_matrices) if type(gm) is CategoricalGroupMatrix]
    gm = dm.group_matrices[g]
    labels = rng.integers(0, 12, n)
    labels[df.f.to_numpy() == "c"] = 4
    labels[gm.codes == gm.n_levels] = 9
    nb = from_groups(labels, n)
    active = np.ones(n, dtype=bool)
    found = unidentified_levels(nb, dm, model._groups, active, penalised)
    columns = np.column_stack([np.ones(n), gm.toarray()])
    deficient = [
        k
        for k in range(nb.m)
        if np.linalg.matrix_rank(np.delete(columns, nb.drop(k), axis=0)) < columns.shape[1]
    ]
    assert sorted(c.neighbourhood for c in found) == deficient == [4, 9]
    assert {c.term for c in found} == {model._groups[g].name}
    assert [c.reference for c in found if c.neighbourhood == 9] == [True]
