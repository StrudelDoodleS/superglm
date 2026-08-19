"""Adversarial tests for build-time interaction cell pruning.

``CategoricalInteraction.build`` prunes two kinds of cell from the non-base
grid before the block is emitted:

* **empty** cells -- no positive-weight training row occupies them, so their
  columns are identically zero in the weighted geometry;
* one **aliased** cell per exactly nested parent level -- when every
  positive-weight row at a non-base level of one parent sits on the other
  parent's non-base grid, that level's main-effect indicator equals the sum of
  its occupied cells row for row.

Both claims are about *identifiability*, and both are only worth anything if
pruning cannot remove a direction the fit could actually have used.  This file
attacks that, rather than re-confirming it.  Every test here is written to
FAIL if the patch is wrong; the ones that pass are recorded attacks that the
patch survived.

The reference the tests measure against is the **unpruned** build, restored at
runtime by :func:`unpruned_interaction`.  Comparing pruned against unpruned in
one process is the only comparison that isolates pruning from every other
difference between two checkouts.
"""

from __future__ import annotations

import contextlib
import itertools
import pickle

import numpy as np
import pandas as pd
import pytest

import superglm.features.interaction as interaction_mod
from superglm import Adaptive, GroupLasso, SuperGLM
from superglm.features.categorical import Categorical
from superglm.features.interaction import CategoricalInteraction
from superglm.types import GroupInfo

# ── the unpruned reference build ──────────────────────────────────

_PRUNED_BUILD = CategoricalInteraction.build


def _unpruned_build(self, x_cat1, x_cat2, parent_specs, sample_weight=None, alias_prune=True):
    """The pre-pruning contract: every non-base cell gets a column.

    Runs the real build first so all level bookkeeping (bases, non-base sets,
    grouping resolution, label validation) is byte-identical, then restores
    the full grid and re-derives the codes against it.  ``alias_prune`` is
    accepted and forwarded so the builder can call this exactly as it calls
    the real build; it makes no difference here, since the full grid is
    restored either way.
    """
    info = _PRUNED_BUILD(
        self, x_cat1, x_cat2, parent_specs, sample_weight=sample_weight, alias_prune=alias_prune
    )
    if getattr(self, "_grid_to_col", None) is None:
        return info
    labels1 = interaction_mod._categorical_build_labels(
        x_cat1, parent_specs[self.cat1_name], context=self.cat1_name
    )
    labels2 = interaction_mod._categorical_build_labels(
        x_cat2, parent_specs[self.cat2_name], context=self.cat2_name
    )
    self._pairs = list(self._all_pairs)
    self._pruned_pairs = []
    self._grid_to_col = None
    codes = interaction_mod._pair_codes(labels1, labels2, self._non_base1, self._non_base2)
    return GroupInfo(columns=None, n_cols=len(self._pairs), cat_codes=codes)


@contextlib.contextmanager
def unpruned_interaction():
    """Run the enclosed fit with cell pruning switched off."""
    CategoricalInteraction.build = _unpruned_build
    try:
        yield
    finally:
        CategoricalInteraction.build = _PRUNED_BUILD


# ── small helpers ─────────────────────────────────────────────────


def _build_block(x1, x2, sample_weight=None):
    cat1 = Categorical(base="first")
    cat2 = Categorical(base="first")
    cat1.build(x1, sample_weight=sample_weight)
    cat2.build(x2, sample_weight=sample_weight)
    ci = CategoricalInteraction("c1", "c2")
    info = ci.build(x1, x2, {"c1": cat1, "c2": cat2}, sample_weight=sample_weight)
    return ci, info


def _designs(x1, x2, ci):
    """``(full, kept)`` designs: intercept + both main effects + cells.

    *full* carries every non-base cell, *kept* only the emitted ones.  These
    are the two column spaces whose equality is the whole safety claim.
    """
    nb1, nb2 = ci._non_base1, ci._non_base2
    n = len(x1)
    main1 = np.column_stack([(x1 == lev).astype(float) for lev in nb1])
    main2 = np.column_stack([(x2 == lev).astype(float) for lev in nb2])
    cells = np.zeros((n, len(nb1) * len(nb2)))
    for i, lev1 in enumerate(nb1):
        for j, lev2 in enumerate(nb2):
            cells[:, i * len(nb2) + j] = ((x1 == lev1) & (x2 == lev2)).astype(float)
    full = np.column_stack([np.ones(n), main1, main2, cells])
    kept = np.column_stack([np.ones(n), main1, main2, ci.transform(x1, x2)])
    return full, kept


def _column_space(X):
    """Orthonormal basis of ``col(X)`` and its numerical rank."""
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    tol = max(X.shape) * np.finfo(float).eps * (s[0] if s.size else 0.0)
    keep = s > tol
    return U[:, keep], int(keep.sum())


def _grid_from_pattern(occ):
    """One row per occupied cell of a level table, repeated so n > p."""
    rows = [
        (f"A{i}", f"B{j}") for i in range(occ.shape[0]) for j in range(occ.shape[1]) if occ[i, j]
    ]
    if len(rows) < 2:
        return None
    x1 = np.array([r[0] for r in rows] * 3)
    x2 = np.array([r[1] for r in rows] * 3)
    return x1, x2


def _occupied_cells(ci, x1, x2):
    return {(a, b) for a, b in ci._all_pairs if ((x1 == a) & (x2 == b)).any()}


def _alias_drops(ci, x1, x2):
    """Pruned cells that were OCCUPIED -- i.e. dropped by the alias rule."""
    occupied = _occupied_cells(ci, x1, x2)
    return [p for p in ci._pruned_pairs if p in occupied]


def _frame(cells, reps=40, seed=1):
    lev1, lev2 = [], []
    for a, b in cells:
        lev1 += [a] * reps
        lev2 += [b] * reps
    X = pd.DataFrame({"c1": lev1, "c2": lev2})
    y = np.random.default_rng(seed).gamma(2.0, 1.5, len(X))
    return X, y


def _fit(X, y, sample_weight=None, **kwargs):
    model = SuperGLM(
        features={"c1": Categorical(base="first"), "c2": Categorical(base="first")},
        interactions=[("c1", "c2")],
        **kwargs,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def _ab(X, y, sample_weight=None, score_on=None, **kwargs):
    """Fit the same specification with and without pruning."""
    pruned = _fit(X, y, sample_weight=sample_weight, **kwargs)
    with unpruned_interaction():
        unpruned = _fit(X, y, sample_weight=sample_weight, **kwargs)
    target = X if score_on is None else score_on
    return pruned, unpruned, pruned.predict(target), unpruned.predict(target)


# Level tables used repeatedly.  Base levels are "A"/"X" (base="first").
EMPTY_ONLY = [
    ("A", "X"),
    ("A", "Y"),
    ("A", "Z"),
    ("B", "X"),
    ("B", "Y"),
    ("B", "Z"),
    ("C", "X"),
    ("C", "Y"),
]  # (C,Z) never occurs
ALIAS_ONLY = [
    ("A", "X"),
    ("A", "Y"),
    ("A", "Z"),
    ("B", "X"),
    ("B", "Y"),
    ("B", "Z"),
    ("C", "Y"),
    ("C", "Z"),
]  # C nested: never at base X
BOTH = [
    ("A", "X"),
    ("A", "Y"),
    ("A", "Z"),
    ("B", "X"),
    ("B", "Y"),
    ("B", "Z"),
    ("C", "Y"),
    ("D", "Y"),
    ("D", "Z"),
]  # (C,Z) empty AND C, D nested


# ══ Attack 1: does the second prune loop over-prune? ══════════════
#
# Both loops compute their candidate from the UNMODIFIED ``grid_occupied``
# and write into the same ``drop`` array, so nothing stops them dropping two
# different cells for two circuits that share a cell.  If the two drops are
# not simultaneously removable, the block loses an identifiable direction and
# the fit changes.  These tests search for that grid.


class TestOverPruning:
    def _scan(self, patterns, shape):
        """Assert span preservation over a family of level tables.

        Returns coverage counters so the caller can prove the scan was not
        vacuous.
        """
        n_pruned = n_alias = n_multi_alias = 0
        failures = []
        for pattern in patterns:
            occ = np.asarray(pattern, dtype=int).reshape(shape)
            grid = _grid_from_pattern(occ)
            if grid is None:
                continue
            x1, x2 = grid
            try:
                ci, _ = _build_block(x1, x2)
            except ValueError:
                continue  # a parent with < 2 levels: not a pruning question
            if ci._grid_to_col is None:
                continue
            n_pruned += 1
            drops = _alias_drops(ci, x1, x2)
            n_alias += bool(drops)
            n_multi_alias += len(drops) >= 2
            full, kept = _designs(x1, x2, ci)
            basis_full, rank_full = _column_space(full)
            basis_kept, rank_kept = _column_space(kept)
            if rank_full != rank_kept:
                failures.append((occ.tolist(), rank_full, rank_kept, list(ci._pruned_pairs)))
                continue
            # Rank alone is weak: assert the SPACES coincide, by projecting
            # each full-design basis vector onto the kept space.
            residual = basis_full - basis_kept @ (basis_kept.T @ basis_full)
            if np.abs(residual).max() > 1e-9:
                failures.append(
                    (occ.tolist(), "span", float(np.abs(residual).max()), list(ci._pruned_pairs))
                )
        assert not failures, (
            f"pruning changed the column space on {len(failures)} grid(s): {failures[:5]}"
        )
        return n_pruned, n_alias, n_multi_alias

    def test_every_three_by_three_level_table_preserves_the_column_space(self):
        patterns = list(itertools.product([0, 1], repeat=9))
        n_pruned, n_alias, n_multi = self._scan(patterns, (3, 3))
        # Coverage: the scan is worthless if it never reached the two-loop case.
        assert n_pruned > 100, n_pruned
        assert n_alias > 50, n_alias
        assert n_multi > 10, f"scan never exercised two simultaneous alias drops ({n_multi})"

    @pytest.mark.parametrize("shape,count", [((4, 4), 400), ((5, 4), 300)])
    def test_random_larger_level_tables_preserve_the_column_space(self, shape, count):
        rng = np.random.default_rng(20260819)
        patterns = [rng.integers(0, 2, shape[0] * shape[1]) for _ in range(count)]
        n_pruned, n_alias, n_multi = self._scan(patterns, shape)
        assert n_multi > 50, f"scan never exercised two simultaneous alias drops ({n_multi})"

    def test_alias_drops_never_close_a_cycle_in_the_circuit_graph(self):
        """The mechanism that makes over-pruning unconstructible.

        Each nested row and each nested column is one aliasing circuit; the
        cell dropped for it is an EDGE between the two circuits it belongs to.
        Dropping one column per circuit is safe exactly while those edges form
        a forest -- a cycle would mean k dropped columns against only k-1
        independent dependencies, i.e. a lost direction.

        The graph is bipartite (row circuits on one side, column circuits on
        the other), so it has no odd cycles, and an even cycle is ruled out by
        the tie-break being EXTREMAL and applied with one consistent
        orientation to every circuit in a loop.  Walk a cycle
        ``r1 - c1 - r2 - c2 - ... - r1``.  Its row-loop edges say ``c1`` is the
        last occupied column of row ``r1`` while ``(r1, c2)`` is occupied, so
        ``c2 <= c1``; the next edge says ``c1 <= c2``; going round gives
        ``c1 = c2 = ...`` and the "cycle" collapses to one edge.  (Mutating the
        rule to ``cells[0]`` merely flips every inequality, so the argument --
        and, measured, the safety -- does not depend on last-versus-first.
        What it depends on is one rule per loop.)  This test asserts the
        conclusion on every 3x3 and 400 random 4x4 tables.
        """
        rng = np.random.default_rng(7)
        patterns = [(np.asarray(p), (3, 3)) for p in itertools.product([0, 1], repeat=9)]
        patterns += [(rng.integers(0, 2, 16), (4, 4)) for _ in range(400)]
        checked = 0
        for pattern, shape in patterns:
            occ = np.asarray(pattern, dtype=int).reshape(shape)
            grid = _grid_from_pattern(occ)
            if grid is None:
                continue
            x1, x2 = grid
            try:
                ci, _ = _build_block(x1, x2)
            except ValueError:
                continue
            drops = _alias_drops(ci, x1, x2)
            if len(drops) < 2:
                continue
            checked += 1
            nb1, nb2 = ci._non_base1, ci._non_base2
            nested_rows = {
                lev for lev in nb1 if (x1 == lev).any() and np.all(np.isin(x2[x1 == lev], nb2))
            }
            nested_cols = {
                lev for lev in nb2 if (x2 == lev).any() and np.all(np.isin(x1[x2 == lev], nb1))
            }
            parent: dict = {}

            def find(v):
                parent.setdefault(v, v)
                while parent[v] != v:
                    parent[v] = parent[parent[v]]
                    v = parent[v]
                return v

            for lev1, lev2 in drops:
                a = ("row", lev1) if lev1 in nested_rows else None
                b = ("col", lev2) if lev2 in nested_cols else None
                if a is None or b is None:
                    continue  # a pendant edge can never close a cycle
                ra, rb = find(a), find(b)
                assert ra != rb, f"alias drops close a cycle on table {occ.tolist()}: {drops}"
                parent[ra] = rb
        assert checked > 100, f"cycle scan never saw a multi-drop grid ({checked})"

    def test_a_four_cycle_of_alias_drops_cannot_be_constructed(self):
        """Direct search for the two-loop counterexample, cell by cell.

        Over every 4x4 level table, look for two nested rows and two nested
        columns whose four dropped cells form the 4-cycle
        ``(r1,c1) (r2,c1) (r2,c2) (r1,c2)``.  That configuration is the
        minimal over-prune.  Finding one would be the defect; finding none
        pins the tie-break as the thing that prevents it.
        """
        rng = np.random.default_rng(11)
        found = []
        for pattern in [rng.integers(0, 2, 16) for _ in range(2000)]:
            occ = np.asarray(pattern, dtype=int).reshape(4, 4)
            grid = _grid_from_pattern(occ)
            if grid is None:
                continue
            x1, x2 = grid
            try:
                ci, _ = _build_block(x1, x2)
            except ValueError:
                continue
            drops = set(_alias_drops(ci, x1, x2))
            for (r1, c1), (r2, c2) in itertools.permutations(drops, 2):
                if r1 != r2 and c1 != c2 and (r2, c1) in drops and (r1, c2) in drops:
                    found.append((occ.tolist(), sorted(drops)))
        assert not found, f"four-cycle over-prune found: {found[:3]}"


# ══ Attack 2: is routing a pruned cell to -1 right for an ALIAS drop? ══
#
# An empty cell's column is identically zero, so its coefficient is zero
# under any objective.  An alias-dropped cell's column is NOT zero -- it is
# merely redundant -- so "contributes zero" is a claim about the SOLVER, not
# about the design.  These tests separate the two.


class TestAliasDropVersusEmptyDrop:
    @pytest.mark.parametrize(
        "cells,expect_alias", [(EMPTY_ONLY, False), (ALIAS_ONLY, True), (BOTH, True)]
    )
    @pytest.mark.parametrize("family", ["gaussian", "poisson"])
    def test_pruned_fit_equals_the_unpruned_fit(self, cells, expect_alias, family):
        X, y = _frame(cells)
        pruned, unpruned, p_pruned, p_unpruned = _ab(X, y, family=family, selection_penalty=0.0)
        spec = pruned._interaction_specs["c1:c2"]
        assert spec._pruned_pairs, "fixture stopped exercising pruning"
        occupied = _occupied_cells(spec, np.asarray(X.c1), np.asarray(X.c2))
        alias = [p for p in spec._pruned_pairs if p in occupied]
        assert bool(alias) is expect_alias, (spec._pruned_pairs, alias)
        assert len(unpruned._interaction_specs["c1:c2"]._pairs) > len(spec._pairs)
        np.testing.assert_allclose(p_pruned, p_unpruned, rtol=1e-10, atol=1e-12)

    def test_rows_in_an_alias_dropped_cell_score_like_the_unpruned_fit(self):
        """The distinguishing case: score rows that LAND in the dropped cell.

        For an empty cell there are no such training rows, so the question
        only has content for an alias drop.
        """
        X, y = _frame(ALIAS_ONLY)
        pruned, unpruned, _, _ = _ab(X, y, family="poisson", selection_penalty=0.0)
        spec = pruned._interaction_specs["c1:c2"]
        dropped = spec._pruned_pairs[0]
        rows = np.asarray((X.c1 == dropped[0]) & (X.c2 == dropped[1]))
        assert rows.sum() > 0, "the alias-dropped cell must carry training rows"
        np.testing.assert_allclose(
            pruned.predict(X)[rows], unpruned.predict(X)[rows], rtol=1e-10, atol=1e-12
        )

    def test_scoring_a_cell_never_seen_in_training_matches(self):
        X, y = _frame(EMPTY_ONLY)
        new = pd.DataFrame({"c1": ["A", "B", "C", "C"], "c2": ["X", "Y", "Z", "X"]})
        _, _, p_pruned, p_unpruned = _ab(
            X, y, score_on=new, family="poisson", selection_penalty=0.0
        )
        np.testing.assert_allclose(p_pruned, p_unpruned, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize(
        "cells",
        [
            ALIAS_ONLY,
            # Two nested rows -> two independent circuits, both resolved.
            [("A", "X"), ("A", "Y"), ("A", "Z"), ("B", "Y"), ("B", "Z"), ("C", "Y"), ("C", "Z")],
            # A nested row AND a nested column, sharing no cell.
            [("A", "X"), ("A", "Y"), ("B", "X"), ("B", "Y"), ("B", "Z"), ("C", "Y"), ("C", "Z")],
        ],
    )
    def test_the_dropped_cell_is_the_one_the_unpruned_solver_zeroes(self, cells):
        """The claim ``reconstruct`` rests on when it reports a pruned cell as 0.

        Build-time pruning is only a relabelling of the same fit if the column
        it removes is the column the rank machinery would have zeroed anyway.
        Anything else still predicts identically -- any valid drop does -- but
        the reported grid would then attribute the level's effect to the wrong
        cell.  This checks it end to end against a live unpruned fit instead
        of by reading the representative rule.
        """
        X, y = _frame(cells)
        pruned = _fit(X, y, family="gaussian", selection_penalty=0.0)
        with unpruned_interaction():
            unpruned = _fit(X, y, family="gaussian", selection_penalty=0.0)
        spec_p = pruned._interaction_specs["c1:c2"]
        spec_u = unpruned._interaction_specs["c1:c2"]
        group = next(g for g in unpruned._groups if g.feature_name == "c1:c2")
        beta_u = unpruned.result.beta[group.sl]
        scale = max(1e-12, float(np.abs(beta_u).max()))
        zeroed = {
            spec_u._pairs[i] for i in range(len(spec_u._pairs)) if abs(beta_u[i]) <= 1e-9 * scale
        }
        occupied = _occupied_cells(spec_p, np.asarray(X.c1), np.asarray(X.c2))
        alias_drops = {p for p in spec_p._pruned_pairs if p in occupied}
        assert alias_drops, spec_p._pruned_pairs
        assert alias_drops <= zeroed, (
            f"pruning dropped {sorted(alias_drops - zeroed)}, but the unpruned solver "
            f"zeroed {sorted(zeroed)} -- the reported grid would attribute the effect "
            "to a different cell"
        )

    # Only grids with an EMPTY cell still prune while a selection penalty
    # suppresses the alias half; an alias-only grid emits its full width there.
    @pytest.mark.parametrize("cells", [EMPTY_ONLY, BOTH])
    def test_group_lasso_weight_follows_the_spanned_width_not_the_emitted_one(self, cells):
        """The group must be priced at the width it SPANS.

        ``GroupSlice.weight`` is ``sqrt(p_g)`` and the group penalty is
        ``lambda1 * weight * ||beta_g||``.  If ``p_g`` followed the emitted
        width, pruning would narrow the group and the same lambda1 would buy
        less shrinkage -- a change in the fit, not a reparametrisation.  The
        pruned columns are structurally unidentifiable, so the block still
        spans the full non-base grid and is priced at it.
        """
        X, y = _frame(cells)
        pruned = _fit(X, y, family="gaussian", selection_penalty=0.5)
        with unpruned_interaction():
            unpruned = _fit(X, y, family="gaussian", selection_penalty=0.5)
        g_pruned = next(g for g in pruned._groups if g.feature_name == "c1:c2")
        g_unpruned = next(g for g in unpruned._groups if g.feature_name == "c1:c2")
        spec = pruned._interaction_specs["c1:c2"]
        assert spec._pruned_pairs, "fixture no longer exercises pruning"
        # The emitted block really is narrower -- this is not a vacuous pass.
        assert g_pruned.size < g_unpruned.size
        assert g_pruned.penalty_size == g_unpruned.penalty_size == len(spec._all_pairs)
        assert g_pruned.weight == pytest.approx(np.sqrt(len(spec._all_pairs)))
        assert g_pruned.weight == pytest.approx(g_unpruned.weight)

    def test_selection_penalty_that_exempts_the_interaction_is_exact(self):
        """Pruning is exact at any lambda1 while the block carries no group penalty."""
        X, y = _frame(EMPTY_ONLY)
        _, _, p_pruned, p_unpruned = _ab(
            X, y, family="gaussian", selection_penalty=0.5, penalty_features=["c1", "c2"]
        )
        np.testing.assert_allclose(p_pruned, p_unpruned, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("cells", [EMPTY_ONLY, ALIAS_ONLY])
    def test_pruning_is_fit_invariant_under_a_selection_penalty(self, cells):
        X, y = _frame(cells)
        _, _, p_pruned, p_unpruned = _ab(X, y, family="gaussian", selection_penalty=0.5)
        np.testing.assert_allclose(p_pruned, p_unpruned, rtol=1e-9, atol=1e-12)

    def test_the_alias_prune_stands_down_under_a_selection_penalty(self):
        """The gate itself, stated so the invariance results cannot go vacuous.

        Dropping an aliased cell picks one representative of a rank deficiency
        spanning the interaction and its parent main effect.  That is free
        while the solver's rank convention picks it, and it is NOT free under
        a group penalty, which picks the representative minimising the group
        norms instead.  So the alias half stands down and the empty half --
        whose columns are identically zero under every penalty -- does not.
        """
        X, y = _frame(ALIAS_ONLY)
        unpenalized = _fit(X, y, family="gaussian", selection_penalty=0.0)
        penalized = _fit(X, y, family="gaussian", selection_penalty=0.5)
        assert unpenalized._interaction_specs["c1:c2"]._pruned_pairs == [("C", "Z")]
        assert penalized._interaction_specs["c1:c2"]._pruned_pairs == []
        assert penalized._interaction_specs["c1:c2"]._grid_to_col is None

        # ...but an EMPTY cell goes in both, because nothing can want it back.
        Xb, yb = _frame(BOTH)
        both_penalized = _fit(Xb, yb, family="gaussian", selection_penalty=0.5)
        spec = both_penalized._interaction_specs["c1:c2"]
        assert spec._pruned_pairs == [("C", "Z")], spec._pruned_pairs
        occupied = _occupied_cells(spec, np.asarray(Xb.c1), np.asarray(Xb.c2))
        assert ("C", "Z") not in occupied, "this must be the EMPTY drop"

    def test_auto_selection_penalty_also_stands_the_alias_prune_down(self):
        """``"auto"`` is not calibrated until the design exists, so the gate
        has to read the intent rather than the resolved value."""
        X, y = _frame(ALIAS_ONLY)
        model = _fit(X, y, family="gaussian", selection_penalty="auto")
        # The configured intent is still the string; only the fit resolves it.
        assert model.penalty.lambda1 == "auto"
        assert model.selection_penalty_ > 0.0
        assert model._interaction_specs["c1:c2"]._pruned_pairs == []

    def test_fit_path_stands_the_alias_prune_down_whatever_it_was_configured_with(self):
        """A path builds ONE design and sweeps lambda1 over a positive grid.

        The configured value says nothing about the grid, so a path must build
        the design a positive lambda1 requires even when it starts from zero.
        """
        X, y = _frame(ALIAS_ONLY)
        model = SuperGLM(
            features={"c1": Categorical(base="first"), "c2": Categorical(base="first")},
            interactions=[("c1", "c2")],
            selection_penalty=0.0,
        )
        model.fit_path(X, y, n_lambda=4)
        assert model._interaction_specs["c1:c2"]._pruned_pairs == []

    def test_no_selection_penalty_gap_at_any_lambda1(self):
        """The gap this file used to pin, across the lambda1 range that showed it.

        It ran 1.2e-4 relative at lambda1=0.05 and rose monotonically to
        5.7e-3 at lambda1=3.2 while ``sqrt(p_g)`` followed the emitted width.
        Every one of those must now be at round-off.
        """
        X, y = _frame(EMPTY_ONLY)
        for lam in (0.05, 0.5, 3.2):
            _, _, p_pruned, p_unpruned = _ab(X, y, family="gaussian", selection_penalty=lam)
            gap = np.abs(p_pruned - p_unpruned).max() / np.abs(p_unpruned).max()
            assert gap < 1e-9, (lam, gap)

    # Only grids with an EMPTY cell still prune while a selection penalty
    # suppresses the alias half; an alias-only grid emits its full width there.
    @pytest.mark.parametrize("cells", [EMPTY_ONLY, BOTH])
    def test_auto_selection_penalty_resolves_to_the_same_lambda1(self, cells):
        """``selection_penalty="auto"`` calibrates off lambda_max, which reads
        the group weights.  A weight that followed the emitted width would move
        the whole path, not just the shrinkage at one lambda1."""
        X, y = _frame(cells)
        pruned, unpruned, p_pruned, p_unpruned = _ab(
            X, y, family="gaussian", selection_penalty="auto"
        )
        assert pruned._interaction_specs["c1:c2"]._pruned_pairs
        np.testing.assert_allclose(p_pruned, p_unpruned, rtol=1e-9, atol=1e-12)

    # Only grids with an EMPTY cell still prune while a selection penalty
    # suppresses the alias half; an alias-only grid emits its full width there.
    @pytest.mark.parametrize("cells", [EMPTY_ONLY, BOTH])
    def test_the_df_ledger_is_invariant_under_a_selection_penalty(self, cells):
        """The group weight enters the df ledger, not just the coefficients.

        A group-lasso lambda1 contributes local curvature ``lambda1 * weight``
        to the inference geometry, so the hat trace reads the weight directly.
        A weight that followed the emitted width would move ``effective_df``
        and with it ``phi``, hence every standard error and every information
        criterion -- on a change that is supposed to be a reparametrisation.
        At lambda1=0 there is no such curvature and both arms already agree,
        which is why this is pinned at lambda1 > 0.
        """
        X, y = _frame(cells)
        pruned = _fit(X, y, family="gaussian", selection_penalty=0.5)
        with unpruned_interaction():
            unpruned = _fit(X, y, family="gaussian", selection_penalty=0.5)
        assert pruned._interaction_specs["c1:c2"]._pruned_pairs
        assert pruned.result.effective_df == pytest.approx(unpruned.result.effective_df, rel=1e-9)
        assert pruned.result.phi == pytest.approx(unpruned.result.phi, rel=1e-9)

    # Only grids with an EMPTY cell still prune while a selection penalty
    # suppresses the alias half; an alias-only grid emits its full width there.
    @pytest.mark.parametrize("cells", [EMPTY_ONLY, BOTH])
    def test_an_adaptive_flavor_reweights_from_the_spanned_width(self, cells):
        """``Adaptive`` re-derives ``sqrt(p_g)`` from the slice it is handed.

        It is the one consumer that does not inherit the constructed weight,
        so it needs the spanned width in its own right; otherwise the fix
        holds for a plain group lasso and silently lapses the moment a caller
        attaches a flavour.
        """
        X, y = _frame(cells)
        # A fresh penalty per arm: the fit resolves lambda1 onto the object.
        pruned = _fit(X, y, family="gaussian", penalty=GroupLasso(lambda1=0.5, flavor=Adaptive()))
        with unpruned_interaction():
            unpruned = _fit(
                X, y, family="gaussian", penalty=GroupLasso(lambda1=0.5, flavor=Adaptive())
            )
        assert pruned._interaction_specs["c1:c2"]._pruned_pairs
        np.testing.assert_allclose(pruned.predict(X), unpruned.predict(X), rtol=1e-9, atol=1e-12)


# ══ Attack 3: sample_weight ═══════════════════════════════════════


class TestSampleWeight:
    def test_none_and_unit_weights_prune_identically(self):
        x1 = np.array([a for a, _ in EMPTY_ONLY] * 5)
        x2 = np.array([b for _, b in EMPTY_ONLY] * 5)
        none_ci, none_info = _build_block(x1, x2, sample_weight=None)
        ones_ci, ones_info = _build_block(x1, x2, sample_weight=np.ones(len(x1)))
        assert none_info.n_cols == ones_info.n_cols
        assert none_ci._pairs == ones_ci._pairs
        assert none_ci._pruned_pairs == ones_ci._pruned_pairs

    def test_a_cell_held_open_only_by_zero_weight_rows_is_pruned_and_exact(self):
        """Zero-weight rows must not keep a structurally dead cell alive."""
        cells = [(a, b) for a in "ABC" for b in "XYZ"]
        X, y = _frame(cells)
        w = np.ones(len(X))
        w[np.asarray((X.c1 == "C") & (X.c2 == "Z"))] = 0.0
        pruned, unpruned, p_pruned, p_unpruned = _ab(
            X, y, sample_weight=w, family="gaussian", selection_penalty=0.0
        )
        assert ("C", "Z") in pruned._interaction_specs["c1:c2"]._pruned_pairs
        np.testing.assert_allclose(p_pruned, p_unpruned, rtol=1e-10, atol=1e-12)

    def test_zero_weight_rows_that_create_the_nesting_are_exact(self):
        """The sharp case: level C is nested only once zero-weight rows are dropped.

        (C, base) rows exist but carry no weight, so the alias identity holds
        in the WEIGHTED geometry the solver sees and nowhere else.  Pruning on
        that basis is only safe if the unpruned solver agrees.
        """
        cells = [(a, b) for a in "ABC" for b in "XYZ"]
        X, y = _frame(cells)
        w = np.ones(len(X))
        w[np.asarray((X.c1 == "C") & (X.c2 == "X"))] = 0.0
        pruned, unpruned, p_pruned, p_unpruned = _ab(
            X, y, sample_weight=w, family="gaussian", selection_penalty=0.0
        )
        spec = pruned._interaction_specs["c1:c2"]
        assert spec._pruned_pairs == [("C", "Z")], spec._pruned_pairs
        occupied = _occupied_cells(spec, np.asarray(X.c1), np.asarray(X.c2))
        assert ("C", "Z") in occupied, "this must be an ALIAS drop, not an empty one"
        np.testing.assert_allclose(p_pruned, p_unpruned, rtol=1e-10, atol=1e-12)

    def test_zero_weight_rows_still_predict_through_a_pruned_cell(self):
        cells = [(a, b) for a in "ABC" for b in "XYZ"]
        X, y = _frame(cells)
        w = np.ones(len(X))
        w[np.asarray((X.c1 == "C") & (X.c2 == "Z"))] = 0.0
        _, _, p_pruned, p_unpruned = _ab(
            X, y, sample_weight=w, family="poisson", selection_penalty=0.0
        )
        rows = np.asarray((X.c1 == "C") & (X.c2 == "Z"))
        np.testing.assert_allclose(p_pruned[rows], p_unpruned[rows], rtol=1e-10, atol=1e-12)

    def test_non_uniform_weights_decide_occupancy_by_weight_not_by_row_count(self):
        x1 = np.array(["A", "A", "A", "B", "B", "B"])
        x2 = np.array(["X", "Y", "Z", "X", "Y", "Z"])
        w = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 3.0])
        ci, info = _build_block(x1, x2, sample_weight=w)
        assert ("B", "Y") in ci._pruned_pairs
        assert ("B", "Z") in ci._pairs or ("B", "Z") in ci._pruned_pairs
        assert info.n_cols == len(ci._pairs)


# ══ Attack 4: does reconstruct() keep the full-grid shape? ════════


class TestReconstructAndDownstream:
    def _fitted(self, cells=BOTH, **kwargs):
        X, y = _frame(cells)
        kwargs.setdefault("family", "poisson")
        kwargs.setdefault("selection_penalty", 0.0)
        return _fit(X, y, **kwargs), X, y

    def test_reconstruct_reports_the_full_grid_with_pruned_cells_at_zero(self):
        model, X, _ = self._fitted()
        spec = model._interaction_specs["c1:c2"]
        assert spec._pruned_pairs
        raw = model.reconstruct_feature("c1:c2")
        assert raw["pairs"] == spec._all_pairs
        assert len(raw["log_relativities"]) == len(spec._all_pairs)
        assert len(raw["relativities"]) == len(spec._all_pairs)
        for lev1, lev2 in spec._pruned_pairs:
            assert raw["log_relativities"][f"{lev1}:{lev2}"] == 0.0
            assert raw["relativities"][f"{lev1}:{lev2}"] == 1.0
        assert raw["pruned_pairs"] == spec._pruned_pairs

    def test_reconstruct_does_not_shift_kept_cells_through_the_remap(self):
        """An off-by-one in ``_grid_to_col`` would show up here and nowhere else.

        Every kept cell must report ITS OWN coefficient, not a neighbour's, so
        the beta handed in is deliberately all-distinct.
        """
        model, _, _ = self._fitted()
        spec = model._interaction_specs["c1:c2"]
        beta = np.arange(1.0, len(spec._pairs) + 1.0) * 1.7
        raw = spec.reconstruct(beta)
        for j, (lev1, lev2) in enumerate(spec._pairs):
            assert raw["log_relativities"][f"{lev1}:{lev2}"] == pytest.approx(beta[j])
        assert sorted(raw["log_relativities"].values()) == pytest.approx(
            sorted([0.0] * len(spec._pruned_pairs) + list(beta))
        )

    def test_reconstructed_grid_matches_what_the_block_actually_scores(self):
        """The cross-repo contract: a table lookup must reproduce the fit.

        A consumer reads one relativity per cell from the reconstructed grid.
        That value has to equal the interaction block's own contribution to
        the linear predictor for a risk in that cell -- including the pruned
        cells, which the table reports as 1.0.
        """
        model, X, _ = self._fitted()
        spec = model._interaction_specs["c1:c2"]
        group = next(g for g in model._groups if g.feature_name == "c1:c2")
        beta = model.result.beta[group.sl]
        raw = spec.reconstruct(beta)
        contribution = spec.score(np.asarray(X.c1), np.asarray(X.c2), beta)
        for lev1, lev2 in spec._all_pairs:
            rows = np.asarray((X.c1 == lev1) & (X.c2 == lev2))
            if not rows.any():
                continue
            expected = raw["log_relativities"][f"{lev1}:{lev2}"]
            np.testing.assert_allclose(contribution[rows], expected, rtol=0, atol=1e-12)
        # Rows at either base level contribute nothing at all.
        base_rows = np.asarray((X.c1 == spec._base1) | (X.c2 == spec._base2))
        np.testing.assert_array_equal(contribution[base_rows], 0.0)

    def test_transform_and_score_agree_after_pruning(self):
        model, X, _ = self._fitted()
        spec = model._interaction_specs["c1:c2"]
        x1, x2 = np.asarray(X.c1), np.asarray(X.c2)
        T = spec.transform(x1, x2)
        assert T.shape == (len(X), len(spec._pairs))
        beta = np.arange(1.0, len(spec._pairs) + 1.0)
        np.testing.assert_allclose(spec.score(x1, x2, beta), T @ beta)
        # No row may light up more than one emitted column.
        assert set(np.unique(T.sum(axis=1))) <= {0.0, 1.0}

    def test_the_coefficient_table_labels_the_emitted_cells_only(self):
        """Recorded shape change: ``summary()`` and ``reconstruct_feature()``
        no longer agree on how many cells the interaction has.

        The coefficient table follows the fitted columns (correctly -- an
        alias-dropped column has no standard error), while the reconstructed
        grid keeps the full table.  Downstream code that joins the two on cell
        label has to tolerate the difference.
        """
        model, _, _ = self._fitted()
        spec = model._interaction_specs["c1:c2"]
        text = str(model.summary())
        labelled = [f"c1:c2[{a}:{b}]" for a, b in spec._pairs]
        for label in labelled:
            assert label in text, label
        for a, b in spec._pruned_pairs:
            assert f"c1:c2[{a}:{b}]" not in text
        assert len(spec._pairs) < len(model.reconstruct_feature("c1:c2")["pairs"])

    def test_pruned_pairs_does_not_say_WHY_a_cell_was_pruned(self):
        """Recorded gap in the reported payload, not a wrong number.

        Both prune kinds surface as a relativity of exactly 1.0 and a
        ``pruned_pairs`` entry, but they mean different things to a downstream
        rate table: an EMPTY cell's 1.0 is a combination the fit never saw
        (an extrapolation the reader should know about), while an ALIAS
        cell's 1.0 is a real, well-populated combination whose whole effect
        was absorbed into the parent main effect.  Nothing in the payload
        separates them, so a consumer cannot flag the first without
        re-deriving occupancy from the data.
        """
        model, X, _ = self._fitted()
        spec = model._interaction_specs["c1:c2"]
        occupied = _occupied_cells(spec, np.asarray(X.c1), np.asarray(X.c2))
        empty_drops = [p for p in spec._pruned_pairs if p not in occupied]
        alias_drops = [p for p in spec._pruned_pairs if p in occupied]
        assert empty_drops and alias_drops, spec._pruned_pairs
        raw = model.reconstruct_feature("c1:c2")
        for lev1, lev2 in empty_drops + alias_drops:
            assert raw["relativities"][f"{lev1}:{lev2}"] == 1.0
        assert set(raw["pruned_pairs"]) == set(empty_drops) | set(alias_drops)
        assert not any(k for k in raw if "empty" in k or "alias" in k), sorted(raw)

    def test_two_training_slices_reconstruct_the_same_grid(self):
        """The patch's own stated invariant, under a CV-shaped split.

        Different folds occupy different cells, so they prune different cells.
        If the reported grid followed the EMITTED columns, every fold would
        produce a differently shaped table and nothing downstream could align
        them.  Both slices here see every level of both parents, so any shape
        difference would come from pruning alone.
        """
        cells_a = [
            ("A", "X"),
            ("A", "Y"),
            ("A", "Z"),
            ("B", "X"),
            ("B", "Y"),
            ("B", "Z"),
            ("C", "X"),
            ("C", "Y"),
        ]  # (C,Z) empty
        cells_b = [
            ("A", "X"),
            ("A", "Y"),
            ("A", "Z"),
            ("B", "X"),
            ("B", "Z"),
            ("C", "X"),
            ("C", "Y"),
            ("C", "Z"),
        ]  # (B,Y) empty
        grids = []
        for cells in (cells_a, cells_b):
            X, y = _frame(cells)
            model = _fit(X, y, family="poisson", selection_penalty=0.0)
            spec = model._interaction_specs["c1:c2"]
            assert spec._pruned_pairs, cells
            grids.append(model.reconstruct_feature("c1:c2")["pairs"])
        assert grids[0] == grids[1] == [(a, b) for a in "BC" for b in "YZ"]

    def test_pickle_roundtrip_predicts_identically(self):
        model, X, _ = self._fitted()
        new = pd.DataFrame({"c1": ["A", "B", "C", "D"], "c2": ["X", "Y", "Z", "Z"]})
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(restored.predict(X), model.predict(X))
        np.testing.assert_array_equal(restored.predict(new), model.predict(new))
        assert (
            restored._interaction_specs["c1:c2"]._pruned_pairs
            == model._interaction_specs["c1:c2"]._pruned_pairs
        )

    def test_reconstruct_on_a_spec_that_predates_pruning_still_works(self):
        """``build`` may not have run on an unpickled pre-patch spec."""
        model, _, _ = self._fitted()
        spec = model._interaction_specs["c1:c2"]
        legacy = pickle.loads(pickle.dumps(spec))
        legacy._pairs = list(legacy._all_pairs)
        for attr in ("_all_pairs", "_pruned_pairs", "_grid_to_col"):
            delattr(legacy, attr)
        raw = legacy.reconstruct(np.arange(1.0, len(legacy._pairs) + 1.0))
        assert raw["pairs"] == legacy._pairs
        assert raw["pruned_pairs"] == []


# ══ Degradation paths the patch documents but does not test ═══════


class TestDegradationPaths:
    def test_the_alias_rule_never_empties_the_block(self):
        x1 = np.array(["A", "A", "B"])
        x2 = np.array(["X", "Y", "Y"])
        ci, info = _build_block(x1, x2)
        assert info.n_cols == 1
        assert ci._pairs == [("B", "Y")]

    def test_a_grid_with_no_occupied_cell_is_left_alone(self):
        x1 = np.array(["A", "A", "B", "B", "A"])
        x2 = np.array(["X", "Y", "X", "X", "Y"])
        ci, info = _build_block(x1, x2)
        assert ci._grid_to_col is None
        assert info.n_cols == len(ci._all_pairs) == 1
        assert ci._pruned_pairs == []

    def test_a_fully_crossed_grid_is_untouched(self):
        x1 = np.array(["A", "B", "C"] * 100)
        x2 = np.array(["X", "Y", "Z", "X"] * 75)
        ci, info = _build_block(x1, x2)
        assert ci._grid_to_col is None
        assert ci._pruned_pairs == []
        assert info.n_cols == 4

    def test_building_twice_gives_the_same_block(self):
        x1 = np.array([a for a, _ in BOTH] * 5)
        x2 = np.array([b for _, b in BOTH] * 5)
        cat1 = Categorical(base="first")
        cat2 = Categorical(base="first")
        cat1.build(x1)
        cat2.build(x2)
        ci = CategoricalInteraction("c1", "c2")
        first = ci.build(x1, x2, {"c1": cat1, "c2": cat2})
        state1 = (list(ci._pairs), list(ci._pruned_pairs), list(ci._all_pairs), first.n_cols)
        second = ci.build(x1, x2, {"c1": cat1, "c2": cat2})
        state2 = (list(ci._pairs), list(ci._pruned_pairs), list(ci._all_pairs), second.n_cols)
        assert state1 == state2
