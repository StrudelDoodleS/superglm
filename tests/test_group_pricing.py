"""The ``group_pricing`` seam (issue #342).

``GroupSlice.weight = sqrt(p_g)`` and the Breheny-Huang df ledger read the
same group dimension.  By default (``group_pricing="rank"``) that dimension
is the group's identifiable rank -- the width the spec emits after
withholding structurally dead interaction cells -- following the group-lasso
literature, which derives ``sqrt(p_g)`` from the df of the group's score
statistic under a standing full-rank assumption (Yuan & Lin 2006; Meier, van
de Geer & Buhlmann 2008; Lounici, Pontil, Tsybakov & van de Geer 2011;
Breheny & Huang 2015 sec. 2.1; Simon & Tibshirani 2012).
``group_pricing="spanned"`` restores the historical pricing at the width the
term spans, under which cell pruning is a pure reparametrisation of the fit;
that contract is pinned in ``test_alias_prune_adversarial.py``.

The fixture grid spans 4 non-base interaction cells and emits 3 (one cell is
empty, hence unconditionally pruned), so rank pricing gives ``sqrt(3)`` where
spanned pricing gives ``sqrt(4)``.
"""

import pickle

import numpy as np
import pytest

from superglm import SuperGLM
from superglm.model.fit_state import ModelConfig
from superglm.penalties.sparse_group_lasso import SparseGroupLasso
from tests.test_alias_prune_adversarial import EMPTY_ONLY, _fit, _frame

EMITTED = 3  # non-base cells that carry data
SPANNED = 4  # non-base cells the term spans; (C, Z) is empty


def _interaction_group(model):
    return next(g for g in model._groups if g.feature_name == "c1:c2")


class TestRankPricingDefault:
    def test_default_prices_the_emitted_width(self):
        """``sqrt(p_g)`` follows the identifiable rank, and the stamp is absent."""
        X, y = _frame(EMPTY_ONLY)
        model = _fit(X, y, family="gaussian", selection_penalty=0.5)
        spec = model._interaction_specs["c1:c2"]
        assert len(spec._all_pairs) == SPANNED and spec._pruned_pairs  # not vacuous
        g = _interaction_group(model)
        assert g.size == EMITTED
        assert g.penalty_dim is None
        assert g.penalty_size == EMITTED
        assert g.weight == pytest.approx(np.sqrt(EMITTED))

    def test_spanned_pricing_restores_the_historical_stamp(self):
        """The seam: ``"spanned"`` reproduces the pre-#342 weight and stamp."""
        X, y = _frame(EMPTY_ONLY)
        model = _fit(X, y, family="gaussian", selection_penalty=0.5, group_pricing="spanned")
        g = _interaction_group(model)
        assert g.size == EMITTED
        assert g.penalty_dim == SPANNED
        assert g.penalty_size == SPANNED
        assert g.weight == pytest.approx(np.sqrt(SPANNED))

    # Measured on this fixture at fd52f5b9 + this change (gaussian,
    # seed 1, reps 40): the max-abs relative prediction gap between rank and
    # spanned pricing.  Spanned pricing equals the historical fit, so this IS
    # the migration delta a pinned-lambda1 user sees on upgrade.
    @pytest.mark.parametrize(
        "lam,expected_gap",
        [(0.05, 1.215e-4), (0.5, 9.670e-4), (3.2, 5.675e-3)],
    )
    def test_rank_pricing_shrinks_the_affected_group_less_by_a_pinned_amount(
        self, lam, expected_gap
    ):
        """Direction AND magnitude of the behaviour change.

        The weight drops from ``sqrt(4)`` to ``sqrt(3)``, so the same
        lambda1 buys less shrinkage on the affected group: its coefficient
        norm rises, never falls.  The size of the resulting prediction shift
        is pinned so a change to either pricing path shows up as a moved
        magnitude, not just a moved sign.
        """
        X, y = _frame(EMPTY_ONLY)
        rank = _fit(X, y, family="gaussian", selection_penalty=lam)
        spanned = _fit(X, y, family="gaussian", selection_penalty=lam, group_pricing="spanned")
        norm_rank = float(np.linalg.norm(rank.result.beta[_interaction_group(rank).sl]))
        norm_spanned = float(np.linalg.norm(spanned.result.beta[_interaction_group(spanned).sl]))
        assert norm_rank > norm_spanned
        p_rank, p_spanned = rank.predict(X), spanned.predict(X)
        gap = float(np.abs(p_rank - p_spanned).max() / np.abs(p_spanned).max())
        assert gap == pytest.approx(expected_gap, rel=0.2)

    def test_bh_df_ledger_prices_the_emitted_width_by_default(self):
        """The second site: the Breheny-Huang fallback df uses the same rank.

        ``SparseGroupLasso`` contributes no inference curvature, so the df
        ledger falls back to ``p_g - (p_g - 1) * shrink``; under the default
        pricing that ``p_g`` is the emitted width, computed from the same
        ``sqrt(p_g)`` weight -- one decision, two consumers.
        """
        X, y = _frame(EMPTY_ONLY)
        model = _fit(X, y, family="gaussian", penalty=SparseGroupLasso(lambda1=0.05, alpha=0.9))
        g = _interaction_group(model)
        assert g.penalty_size == EMITTED
        norm_g = float(np.linalg.norm(model.result.beta[g.sl]))
        shrink = min(1.0, 0.05 * g.weight / norm_g)
        assert 0.0 < shrink < 1.0, shrink
        df_rank_formula = EMITTED - (EMITTED - 1) * shrink
        df_spanned_formula = SPANNED - (SPANNED - 1) * shrink
        assert model._group_edf["c1:c2"] == pytest.approx(df_rank_formula, rel=1e-9)
        assert abs(model._group_edf["c1:c2"] - df_spanned_formula) > 1e-3
        # The per-column ledger still adds back up to the group's df.
        assert float(np.sum(model.result.rank_info.feature_edf[g.sl])) == pytest.approx(
            model._group_edf["c1:c2"], rel=1e-12
        )

    def test_unpenalized_df_arm_reports_one_less_df_per_pruned_cell(self):
        """A group the penalty skips is priced at ``p_g`` outright.

        The fits are identical between modes (the untargeted group's weight
        never enters the solver), so the ledger alone moves: exactly one df
        per pruned cell, and ``phi`` with it.  This is the post-prune
        "225 real columns reported as 253" fiction, closed by default.
        """
        X, y = _frame(EMPTY_ONLY)

        def penalty():
            return SparseGroupLasso(lambda1=0.05, alpha=0.9, features=["c1", "c2"])

        rank = _fit(X, y, family="gaussian", penalty=penalty())
        spanned = _fit(X, y, family="gaussian", penalty=penalty(), group_pricing="spanned")
        np.testing.assert_allclose(rank.predict(X), spanned.predict(X), rtol=0, atol=0)
        assert rank._group_edf["c1:c2"] == pytest.approx(float(EMITTED))
        assert spanned._group_edf["c1:c2"] == pytest.approx(float(SPANNED))
        assert spanned.result.effective_df - rank.result.effective_df == pytest.approx(
            float(SPANNED - EMITTED), rel=1e-12
        )
        # Fewer df on the same Pearson chi-square means a larger residual df
        # denominator, hence a smaller phi.
        assert rank.result.phi < spanned.result.phi

    def test_lambda_max_auto_and_path_anchor_shift_by_exactly_the_weight_ratio(self):
        """When the repriced group sets lambda_max, the whole grid moves.

        With the penalty restricted to the interaction, lambda_max is
        ``||grad_g|| / w_g`` for that one group, so the rank/spanned ratio is
        exactly ``sqrt(SPANNED/EMITTED)`` -- for the ``"auto"`` calibration
        (10% of lambda_max) and for the ``fit_path`` anchor alike.
        """
        X, y = _frame(EMPTY_ONLY)
        ratio = np.sqrt(SPANNED / EMITTED)

        rank = _fit(
            X,
            y,
            family="gaussian",
            selection_penalty="auto",
            penalty_features=["c1:c2"],
        )
        spanned = _fit(
            X,
            y,
            family="gaussian",
            selection_penalty="auto",
            penalty_features=["c1:c2"],
            group_pricing="spanned",
        )
        assert rank._selection_penalty_fitted == pytest.approx(
            spanned._selection_penalty_fitted * ratio, rel=1e-12
        )

        rank_path = _path(X, y)
        spanned_path = _path(X, y, group_pricing="spanned")
        assert rank_path.lambda_seq[0] == pytest.approx(
            spanned_path.lambda_seq[0] * ratio, rel=1e-12
        )

    def test_auto_is_unmoved_while_an_unaffected_group_is_the_argmax(self):
        """On this fixture a main effect sets lambda_max, so ``"auto"`` and a
        path grid do not move at all -- the shift is conditional, not global."""
        X, y = _frame(EMPTY_ONLY)
        rank = _fit(X, y, family="gaussian", selection_penalty="auto")
        spanned = _fit(X, y, family="gaussian", selection_penalty="auto", group_pricing="spanned")
        assert rank._selection_penalty_fitted == spanned._selection_penalty_fitted


def _path(X, y, **kw):
    from superglm.features import Categorical

    model = SuperGLM(
        family="gaussian",
        features={"c1": Categorical(base="first"), "c2": Categorical(base="first")},
        interactions=[("c1", "c2")],
        selection_penalty=0.5,
        penalty_features=["c1:c2"],
        **kw,
    )
    return model.fit_path(X, y, n_lambda=4)


class TestSeamPlumbing:
    def test_invalid_value_raises_at_construction(self):
        with pytest.raises(ValueError, match="group_pricing"):
            SuperGLM(group_pricing="columns")

    def test_the_builder_validates_and_prices_in_one_place(self):
        """Direct callers of the builder get the same validation, and the one
        decision function emits weight dimension and stamp together."""
        from superglm.dm_builder import _priced_group_dimension, _validate_group_pricing
        from superglm.types import GroupInfo

        with pytest.raises(ValueError, match="group_pricing"):
            _validate_group_pricing("columns")
        info = GroupInfo(columns=None, n_cols=EMITTED, penalty_width=SPANNED)
        assert _priced_group_dimension(info, EMITTED, "rank") == (EMITTED, None)
        assert _priced_group_dimension(info, EMITTED, "spanned") == (SPANNED, SPANNED)
        plain = GroupInfo(columns=None, n_cols=EMITTED)
        assert _priced_group_dimension(plain, EMITTED, "rank") == (EMITTED, None)
        assert _priced_group_dimension(plain, EMITTED, "spanned") == (EMITTED, None)

    @pytest.mark.parametrize("value", ["rank", "spanned"])
    def test_clone_unfitted_preserves_the_mode(self, value):
        model = SuperGLM(group_pricing=value)
        assert model.clone_unfitted()._group_pricing == value

    @pytest.mark.parametrize("value", ["rank", "spanned"])
    def test_pickle_roundtrip_preserves_the_mode(self, value):
        X, y = _frame(EMPTY_ONLY)
        model = _fit(X, y, family="gaussian", selection_penalty=0.5, group_pricing=value)
        restored = pickle.loads(pickle.dumps(model))
        assert restored._group_pricing == value
        assert restored._config.group_pricing == value

    def test_pre_field_pickles_restore_spanned_pricing(self):
        """A config pickled before the field existed keeps the behaviour it
        was fitted under, not the new default."""
        config = SuperGLM()._config
        state = dict(config.__dict__)
        assert state.pop("group_pricing") == "rank"
        legacy = ModelConfig.__new__(ModelConfig)
        legacy.__setstate__(state)
        assert legacy.group_pricing == "spanned"
        assert legacy.constructor_kwargs()["group_pricing"] == "spanned"

    def test_a_model_unpickled_from_before_the_field_builds_spanned(self):
        """Refitting a pre-field model reproduces the pricing it recorded.

        Simulates the unpickled state faithfully: neither the instance
        attribute nor the config field exists, so both the materialize path
        (which every fit attempt takes) and the direct-build fallback must
        default to the historical pricing.
        """
        X, y = _frame(EMPTY_ONLY)
        model = _make_model()
        del model._group_pricing
        state = dict(model._config.__dict__)
        state.pop("group_pricing")
        legacy = ModelConfig.__new__(ModelConfig)
        legacy.__setstate__(state)
        model._config = legacy
        model.fit(X, y)
        g = _interaction_group(model)
        assert g.penalty_size == SPANNED
        assert g.weight == pytest.approx(np.sqrt(SPANNED))


def _make_model():
    from superglm.features import Categorical

    return SuperGLM(
        family="gaussian",
        features={"c1": Categorical(base="first"), "c2": Categorical(base="first")},
        interactions=[("c1", "c2")],
        selection_penalty=0.5,
    )
