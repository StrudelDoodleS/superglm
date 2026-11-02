"""PSST interaction screening over a fitted mains model.

PSST (Penalized Smooth Score Test) ranks every candidate pair by how much of
the fitted model's leftover working signal the interaction block the pair
would refit as could absorb at a fixed screening complexity.  One fused O(n)
cell pass per pair, no refits; the confirmatory ``fit_reml`` refit of the
top-ranked pairs is the gate.  The statistic is a ranking score, not a
calibrated p-value.

The ``kind`` column names the interaction class each pair would refit as:
``ti`` (spline x spline), ``spline_cat``, ``cat_cat``, ``numeric_cat`` and
``numeric_numeric``.  Gridded kinds run through the same cell kernels on a
per-margin ``(codes, support size, menu, penalty)`` description; a categorical
margin is a gridded margin whose menu is the (L, L-1) treatment-contrast block
and whose penalty is absent.  A numeric margin never grids: it enters its probe
linearly and becomes a factor (``screening/_pair_factor``), so such a pair is
exact whenever it is computed and a factor too wide for the pair's blocks is
refused with a NaN row.  A spline-mode ``OrderedCategorical`` margin rides the
spline arm on its mapped level scores.

``z`` centres and scales the score with the fixed Gaussian reference mean and
variance (twice the sum of squared shrinkage factors); it does not give
candidates identical null tails.  See the screening guide for measured floors.
A pair with no penalty in its block is evaluated at a single rung: ``edf0``
reports the block's achieved rank and ``lambda0`` is 0.

``max_cells`` bounds allocation, and through ``_within_cubic_budget`` also
time: the probe block's dimension ``k`` enters every rung as a ``(k, k)``
decomposition (pivoted QR plus SVD), so per-pair time grows as ``k^3`` where
allocations grow as ``k^2``.  A block too wide to solve inside the budget is
refused with a NaN row, immediately, since binning cannot shrink a basis.

The statistic is reported on the ``T / phi`` scale, with ``phi`` the mains
fit's Pearson dispersion estimate by default (denominator ``sum(w) - edf``
under ``"frequency"`` weights, positive-weight rows minus ``edf`` under
``"prior"``), so the screen and the published fit share one scale.

Per-feature work (unique support, codes, marginal menus and penalties) is
cached for the whole sweep, so an all-pairs screen builds each feature's
marginal once, at the cost of holding those arrays in memory.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from superglm._frame import as_eager_frame
from superglm.distributions import Tweedie, validate_response
from superglm.features.categorical import (
    Categorical,
    _codes_against,
    _resolve_categorical_labels,
)
from superglm.features.numeric import Numeric
from superglm.features.ordered_categorical import (
    OrderedCategorical,
    resolve_interaction_parent,
)
from superglm.features.piecewise import Piecewise
from superglm.features.polynomial import Polynomial
from superglm.features.spline import _SplineBase
from superglm.screening import (
    numeric_cat_factor,
    numeric_numeric_factor,
    pair_cell_moments,
    pair_design_factor,
    penalized_score_statistic_ladder,
    working_score,
)
from superglm.screening._overlap import tensor_penalty_root
from superglm.screening._structured import (
    _StructuredBudgetExceededError,
    spline_cat_moments,
    structured_ladder,
)
from superglm.solvers.dispersion import (
    model_weight_semantics,
    pearson_residual_degrees_of_freedom,
)

_RESULT_COLUMNS = [
    "feature_a",
    "feature_b",
    "kind",
    "statistic",
    "z",
    "edf0",
    "lambda0",
    "n_cells",
    "approx",
]

# A class of allocation max_cells alone does not bound: one scaling with a
# margin's SUPPORT times the other margin's basis dimension squared, or with a
# tensor width squared.  Two gates charge against this factor: the arrow kernel's
# (n_a, k_s, k_s) stack in _within_structured_cells and the dense
# (k_l*k_r, k_l*k_r) block a gridded pair reduces to.  _within_budget keeps its
# term at 4 only because it decides admissions.  The marginal dimension is
# estimated from the parent spec's k, a guard-grade bound.
_INTERMEDIATE_BUDGET_FACTOR = 4

# Every budget above bounds an ALLOCATION.  Per-pair TIME is cubic in the probe
# block's dimension k (each rung takes a (k, k) decomposition), and the
# allocation budgets alone admit blocks whose solve costs minutes.  The
# ceilings are k^3 <= _CUBIC_BUDGET_FACTOR * max_cells for an unpenalized block
# and the same budget against _PENALIZED_LADDER_COST times the work for a
# penalized one.  The penalized ladder shares ONE simultaneous diagonalization
# of the pencil (V_eff, S) across its rungs, so edf(lambda) and T(lambda) are
# closed forms and its cost is close to an unpenalized block's.  At the default
# max_cells this admits k <= 1709 unpenalized and k <= 1357 penalized.  Raising
# max_cells lifts both.  The constants are dimensional, so they decide which
# pairs are admitted; changing them changes scope.
_CUBIC_BUDGET_FACTOR = 1000
_PENALIZED_LADDER_COST = 2

# Every constant above budgets a DENSE block.  A spline x categorical pair
# refused by them is retried through the arrow kernel (screening/_structured.py),
# whose cost is linear in the level count.  _within_structured_budget and
# _within_structured_cells bound allocations, which grow as k_s^2, and
# _structured_evaluation_budget limits factor passes using a cubic work proxy.
# The stable profiled-trace setup is deducted from that work ceiling before any
# endpoint evaluations are admitted, and two level-sized stacks (the persistent
# curvature blocks and one additive suffix stack) are charged explicitly.
_STRUCTURED_BUDGET_FACTOR = 2
_STRUCTURED_CUBIC_BUDGET_FACTOR = 50


def _structured_evaluation_allowance(max_cells, n_a, k_s, n_levels):
    """Factor passes left under the dimensional work estimate.

    This is a count of passes, not a certified floating-operation budget.
    Variance passes add contractions and a QR tree to the ordinary pass;
    their cost is measured separately in the reference-variance report.
    """
    max_cells, n_a, k_s, n_levels = (
        int(max_cells),
        int(n_a),
        int(k_s),
        int(n_levels),
    )
    per_evaluation = n_levels * (k_s + 1) ** 3
    setup_work = 2 * n_a * n_levels * k_s**2 + 7 * per_evaluation
    remaining = _STRUCTURED_CUBIC_BUDGET_FACTOR * max_cells - setup_work
    return max(remaining, 0) // max(per_evaluation, 1)


def _quantile_binned(x, bins):
    """Replace x by per-bin mean representatives on an empirical quantile grid.

    Screening-only lossy compression for continuous covariates whose joint
    support would blow the cell budget: the spline basis is evaluated at the
    within-bin mean, so the probe sees the same marginal geometry at <= bins
    support points.  Never applied when the pair fits the budget exactly.
    """
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, bins + 1)[1:-1]))
    codes = np.searchsorted(edges, x, side="right")
    _, codes = np.unique(codes, return_inverse=True)
    reps = np.bincount(codes, weights=x) / np.bincount(codes)
    return reps[codes]


def _validated_budgets(edf0) -> tuple[float, ...]:
    budgets = np.atleast_1d(np.asarray(edf0, dtype=np.float64)).ravel()
    if budgets.size == 0:
        raise ValueError("edf0 must be a positive budget or a non-empty sequence of budgets")
    if not np.all(np.isfinite(budgets)) or np.any(budgets <= 0.0):
        raise ValueError(f"edf0 budgets must be finite and positive, got {edf0!r}")
    return tuple(float(b) for b in budgets)


_DEFERRED_KIND_HINT = (
    "spline x numeric screening is deferred until a varying-coefficient "
    "interaction term exists; respec the Numeric parent as a Spline to screen "
    "the pair as ti(), or see the screening guide. Polynomial margins are "
    "likewise deferred."
)


def _margin_kind(spec) -> str | None:
    """Classify a fitted spec for screening; None means not screenable."""
    if isinstance(spec, _SplineBase):
        return "spline"
    if isinstance(spec, OrderedCategorical):
        # A term with specials= is refused here, before any column is read: a special
        # is a free level with no position on the spline axis, so
        # ``resolve_interaction_parent`` refuses it and would abort the whole sweep.
        if spec.has_specials:
            return None
        # A spline-mode OC is a spline through the level values and screens like one.
        # Step mode has no interaction target, and a Piecewise/Polynomial inner basis is
        # an unpenalized parametric block the interaction classes cannot host.
        if spec.basis == "spline" and spec._spline is not None and spec.basis_kind == "spline":
            return "spline"
        return None
    if isinstance(spec, Categorical):
        # Guard: Categorical.build raises below two levels, so this is unreachable from
        # screen_interactions; it keeps a (1, 0) contrast menu from reaching the kernels.
        return "categorical" if len(spec._levels) >= 2 else None
    if isinstance(spec, Numeric):
        return "numeric"
    return None


def _deferral_reason(spec) -> str:
    """Why a fitted main effect has no screenable margin.

    Called only for names ``_margin_kind`` refused; the reason is reported on
    ``table.attrs["deferred_features"]`` and in the candidates error.
    """
    if isinstance(spec, OrderedCategorical):
        if spec.has_specials:
            return (
                "OrderedCategorical with specials= is deferred: a special is a free "
                "level with no position on the spline axis, so the margin has no "
                "score to grid on; screening the pair needs composite margins"
            )
        # Mirror ``_margin_kind``'s disjunction (``basis != "spline" OR _spline is
        # None``) rather than assuming its complement.
        if spec.basis != "spline":
            return (
                "step-mode OrderedCategorical is deferred: the one-hot geometry, "
                "removed in 0.24.0, has no marginal smooth to cross with"
            )
        if spec._spline is not None and spec.basis_kind != "spline":
            return (
                f"OrderedCategorical(basis={type(spec._basis_spline).__name__}(...)) "
                "margins are deferred: the inner basis is an unpenalized parametric "
                "block, not a penalized marginal smooth, so no interaction class "
                "refits the pair"
            )
        return (
            "OrderedCategorical is deferred: no inner spline was built, so the term "
            "has no marginal smooth to cross with"
        )
    if isinstance(spec, Piecewise):
        return (
            "Piecewise margins are deferred: the hat basis is not a penalized marginal "
            "smooth, so no interaction class refits the pair. A per-level piecewise "
            "(Piecewise x Categorical) is the natural extension and is not built yet"
        )
    if isinstance(spec, Polynomial):
        return (
            "Polynomial margins are deferred: the basis is not a penalized marginal "
            "smooth, so no interaction class refits the pair"
        )
    return f"{type(spec).__name__} margins are deferred: no screenable margin"


# Every entry names a pair kind that has a real refit target; a combination
# absent from this table is DEFERRED, not merely unimplemented, and is dropped
# from the default sweep rather than reported as a null result.
_PAIR_KINDS = {
    frozenset(("spline",)): "ti",
    frozenset(("spline", "categorical")): "spline_cat",
    frozenset(("numeric", "categorical")): "numeric_cat",
    frozenset(("categorical",)): "cat_cat",
    frozenset(("numeric",)): "numeric_numeric",
}


def _pair_kind(kind_a: str, kind_b: str) -> str | None:
    return _PAIR_KINDS.get(frozenset((kind_a, kind_b)))


def _validated_pairs(candidates, margin_kinds, fitted_pairs, deferred_features):
    if candidates is None:
        # A pair the model already fits as an interaction is not a candidate: the
        # confirmation workflow would fail with "interaction already added".
        return [
            pair
            for pair in combinations(margin_kinds, 2)
            if _pair_kind(margin_kinds[pair[0]], margin_kinds[pair[1]]) is not None
            and frozenset(pair) not in fitted_pairs
        ]
    pairs = []
    for raw in candidates:
        pair = tuple(raw)
        if len(pair) == 2 and pair[0] != pair[1]:
            # A name the model fitted but cannot screen is deferred, not a typo; the reason
            # quoted is the one the result table reports.
            deferred_names = sorted((name for name in pair if name in deferred_features), key=repr)
            if deferred_names:
                detail = "; ".join(f"{n} — {deferred_features[n]}" for n in deferred_names)
                raise ValueError(
                    f"candidates entry {raw!r} names fitted feature(s) "
                    f"{deferred_names} that have no screenable margin: {detail}"
                )
        if len(pair) != 2 or pair[0] == pair[1] or not set(margin_kinds).issuperset(pair):
            raise ValueError(
                "candidates entries must pair two distinct screenable fitted "
                f"features; got {raw!r} (screenable features: "
                f"{sorted(margin_kinds, key=repr)})"
            )
        if _pair_kind(margin_kinds[pair[0]], margin_kinds[pair[1]]) is None:
            raise ValueError(
                f"candidates entry {raw!r} pairs kinds "
                f"({margin_kinds[pair[0]]}, {margin_kinds[pair[1]]}) with no "
                f"refit target — {_DEFERRED_KIND_HINT}"
            )
        if frozenset(pair) in fitted_pairs:
            raise ValueError(
                f"candidates entry {raw!r} is already fitted as an interaction; "
                "screening profiles only the parent mains and cannot re-screen it"
            )
        pairs.append(pair)
    return pairs


def _categorical_codes(spec, x_raw) -> tuple[np.ndarray, int]:
    """Dense 0-based codes over the whole level UNIVERSE, in ``spec._levels`` order.

    Applies the fitted spec's grouping collapse and out-of-universe validation,
    so the grid includes the base level and any level pinned to base.
    """
    x = _resolve_categorical_labels(
        x_raw,
        spec._grouping,
        known_levels=set(spec._levels),
    )
    codes = _codes_against(x, spec._levels)
    if codes.size and codes.min() < 0:
        # Defensive: label resolution already rejects anything outside the universe.
        raise ValueError(
            "screen_interactions found categorical values outside the declared "
            f"level universe {list(spec._levels)}"
        )
    return codes, len(spec._levels)


def _contrast_menu(spec) -> np.ndarray:
    """Treatment-contrast menu; base and pinned levels' rows are all zeros.

    Column ``j`` indicates ``spec._non_base[j]``, matching the main effect's
    column order.  Width follows ``_non_base``, not ``L - 1``.
    """
    position = {lev: i for i, lev in enumerate(spec._levels)}
    menu = np.zeros((len(position), len(spec._non_base)), dtype=np.float64)
    for j, lev in enumerate(spec._non_base):
        menu[position[lev], j] = 1.0
    return menu


def _contrast_rows(spec) -> np.ndarray:
    """The one row each contrast column indicates (the one-hot menu as indices).

    Gives the structured kernel ``L - 1`` integers instead of an ``(L, L - 1)``
    block of doubles.
    """
    position = {lev: i for i, lev in enumerate(spec._levels)}
    return np.fromiter(
        (position[lev] for lev in spec._non_base), dtype=np.intp, count=len(spec._non_base)
    )


def _marginal_width_estimate(spec) -> int:
    """Guard-grade marginal width from the parent spline's geometry.

    Takes the margin's effective spec (an OrderedCategorical resolves to its inner
    spline first).  Biased low on purpose: an under-estimate self-heals through the
    post-menu recheck, an over-estimate would wrongly bin or skip a pair.
    ``n_knots`` floors the centred marginal width of every built-in kind.
    Categorical and numeric widths are exact.
    """
    if isinstance(spec, Categorical):
        # _non_base, not L - 1: pinned levels carry no main-effect column, and
        # an over-estimate here skips a pair the budget actually admits.
        return max(len(spec._non_base), 1)
    if isinstance(spec, Numeric):
        return 1
    n_knots = getattr(spec, "n_knots", None)
    if n_knots is not None:
        # n_knots floors every built-in kind including degree-0 ps/bs
        # (centered width n_knots + degree) and minimum-k cr (n_knots + 1);
        # any floor above the true width is a terminal over-estimate.
        return max(int(n_knots), 1)
    return 1


def screen_interactions(
    model,
    X,
    y,
    sample_weight=None,
    *,
    offset=None,
    candidates=None,
    edf0=(2.0, 4.0, 8.0, 16.0),
    max_cells: int = 5_000_000,
    screen_bins: int = 256,
    phi: float | None = None,
) -> pd.DataFrame:
    """Rank candidate pair interactions of a fitted model by PSST.

    Each pair is screened as the interaction class it would refit as, named in
    the ``kind`` column: ``ti`` (spline x spline), ``spline_cat``, ``cat_cat``,
    ``numeric_cat`` (per-level numeric slope) and ``numeric_numeric``.  A
    spline-mode ``OrderedCategorical`` margin screens as a spline on its mapped
    level scores.  A kind whose block carries no penalty is evaluated at a single
    rung, so the ``edf0`` argument does not apply to it.  The ranking does not
    provide calibrated p-values; see the screening guide for measured null
    behaviour and evaluate promising candidates by refit on held-out data.

    ``edf0`` specifies screening complexity.  For candidate shrinkage factors
    ``a`` each rung uses ``z = (T / phi - sum(a)) / sqrt(2 * sum(a**2))``, with
    ``sum(a)`` its achieved EDF, and the pair reports the rung with the largest
    ``z``.  Pass a single float to probe one budget.

    ``offset`` and ``sample_weight`` default to the values the model was fitted
    with (weights only when non-unit).  Inherited arrays are in training row
    order, so inheriting requires ``X``/``y`` to be the retained training data;
    screening a holdout, subsample or reordered frame must pass them explicitly.
    ``T`` is scaled by the fit's Pearson dispersion estimate (module docstring);
    the value used is ``table.attrs["phi"]`` and ``phi=`` overrides it.

    Fitted mains with no screenable margin (``Polynomial``, ``RandomEffect``,
    step-mode ``OrderedCategorical``, any ``OrderedCategorical`` with
    ``specials=``) are excluded and reported in ``table.attrs["deferred_features"]``,
    a ``{feature: reason}`` mapping.  Naming one in ``candidates`` raises with the
    same reason.

    Returns a frame sorted by ``z`` (descending) with columns ``feature_a,
    feature_b, kind, statistic, z, edf0, lambda0, n_cells, approx``;
    ``statistic``/``edf0``/``lambda0`` describe the winning rung, so
    ``statistic`` is not comparable across rows: rank by ``z``.  At a clamped rung
    ``lambda0`` is a bracket edge, not an interpretable smoothing parameter.

    Pairs whose support exceeds the cell or intermediate budgets fall back to
    quantile binning, largest margin first (a margin with more than
    ``screen_bins`` unique values is compressed to ``screen_bins`` quantile bins)
    and the row is flagged ``approx=True``.  Pairs within budget are computed
    exactly, though ``approx`` is also True when the confirmatory refit would
    discretize lossily (a ``ti`` refit bins only when both parents discretize, a
    ``spline_cat`` refit when its spline parent does, and only when a margin's
    cardinality exceeds its bin count).  A pair still over budget after binning,
    or whose statistic degenerates, gets a NaN row (``n_cells`` reports the grid
    attempted); a pair whose tensor curvature block ``(k_a*k_b)^2`` alone exceeds
    the budget is skipped without binning.

    ``max_cells`` bounds allocation and time: ``_within_cubic_budget`` refuses a
    pair whose block is too wide to solve, ``k^3 <= 1000 * max_cells`` unpenalized
    and twice the work penalized (``k <= 1709`` and ``k <= 1357`` at the default).
    A ``spline_cat`` pair refused there is retried through the arrow kernel under
    its own budgets.  Categorical and numeric margins never bin, so ``cat_cat``,
    ``numeric_cat`` and ``numeric_numeric`` rows are always ``approx=False`` and
    their only degradation is refusal: a ``numeric_cat`` pair whose factor is too
    wide to fit ``max_cells`` (2235 levels at the default) or to be solved within
    it (1710 levels) gets a NaN row until ``max_cells`` is raised;
    ``numeric_numeric`` is never refused.  Their ``n_cells`` counts the other
    margin's cells alone (1 when both are numeric).
    """
    if getattr(model, "_result", None) is None:
        raise RuntimeError("screen_interactions requires a fitted model; call fit_reml first")
    from superglm.features.interaction import TensorInteraction, _normalize_tensor_penalty

    frame = as_eager_frame(X)
    n_rows = len(frame)
    y = np.asarray(y, dtype=np.float64)
    weights_inherited = False
    if sample_weight is None:
        # Inherit only a genuinely non-unit fitted weight vector: unit weights cannot
        # mispair rows and inheriting them would pin X/y to the training data.
        # Non-unitness is read from the stored array, not the _fit_used_weights stamp,
        # because the editor may rewrite _fit_weights without touching the stamp.
        stored_weights = getattr(model, "_fit_weights", None)
        if stored_weights is not None:
            if np.any(np.asarray(stored_weights) != 1.0):
                sample_weight = stored_weights
                weights_inherited = True
        elif getattr(model, "_fit_used_weights", False):
            raise ValueError(
                "the model was fitted with non-unit sample_weight but its fit state was "
                "released (retain_fit_state=False); pass sample_weight explicitly"
            )
    weights = (
        np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    )
    if y.shape != (n_rows,):
        raise ValueError(
            f"y must be one-dimensional with one entry per row of X; got shape {y.shape} "
            f"for {n_rows} rows"
        )
    if weights.shape != (n_rows,):
        raise ValueError(
            f"sample_weight must be one-dimensional with one entry per row of X; got shape "
            f"{weights.shape} for {n_rows} rows"
        )
    if not np.all(np.isfinite(y)):
        raise ValueError("screen_interactions requires finite y")
    if (
        isinstance(model._distribution, Tweedie)
        and model_weight_semantics(model) == "prior"
        and (not np.all(np.isfinite(weights)) or np.any(weights <= 0.0))
    ):
        raise ValueError("Tweedie sample_weight must be finite and strictly positive")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or not np.any(weights > 0.0):
        raise ValueError("sample_weight must be finite and non-negative with a positive sum")
    budgets = _validated_budgets(edf0)
    if int(screen_bins) < 2:
        raise ValueError(f"screen_bins must be at least 2, got {screen_bins!r}")
    screen_bins = int(screen_bins)
    if not (np.isfinite(max_cells) and int(max_cells) >= 1):
        raise ValueError(
            f"max_cells is an allocation ceiling and must be a finite positive "
            f"integer, got {max_cells!r}"
        )
    max_cells = int(max_cells)
    if phi is not None and (not np.isfinite(phi) or phi <= 0.0):
        raise ValueError(f"phi override must be finite and positive, got {phi!r}")
    validate_response(y, model._distribution)

    margin_kinds = {
        name: kind
        for name in model._feature_order
        if (kind := _margin_kind(model._specs.get(name))) is not None
    }
    # Every fitted main is either screenable or REPORTED as deferred.  The two
    # sets partition ``_feature_order``, so a caller can tell a feature that
    # screened badly from one that was never screened at all.
    deferred_features = {
        name: _deferral_reason(model._specs.get(name))
        for name in model._feature_order
        if name not in margin_kinds
    }
    # Every interaction class exposes parent_names, so exclusion covers the
    # whole family (tensor, spline_cat, numeric_cat, cat_cat, ..., and
    # FactorSmooth) rather than tensor terms alone.
    fitted_pairs = {
        frozenset(spec.parent_names)
        for spec in getattr(model, "_interaction_specs", {}).values()
        if hasattr(spec, "parent_names")
    }
    pairs = _validated_pairs(candidates, margin_kinds, fitted_pairs, deferred_features)

    def _select_flag(name):
        # Only spline margins carry a double-penalty flag; an OC margin's lives on its
        # inner spline.  Unwrapped rather than resolved, so an unbuildable parent is
        # named before any covariate is read.
        spec = model._specs[name]
        if isinstance(spec, OrderedCategorical):
            spec = spec._spline
        return getattr(spec, "select", False)

    selected = sorted({name for pair in pairs for name in pair if _select_flag(name)}, key=repr)
    if selected:
        raise ValueError(
            f"screen_interactions requires single-penalty parent smooths, but {selected} were "
            "fitted with select=True; ti() terms cannot be built on such parents either — "
            "refit the mains without select"
        )

    raw_x: dict[str, np.ndarray] = {}
    raw_labels: dict[str, np.ndarray] = {}
    margin_source_cache: dict[str, tuple] = {}

    def _raw_numeric(name):
        if name not in raw_x:
            x = frame.column_array(name, dtype=np.float64)
            if not np.all(np.isfinite(x)):
                raise ValueError(
                    f"screen_interactions requires finite covariates; {name!r} contains "
                    "non-finite values"
                )
            raw_x[name] = x
        return raw_x[name]

    def _raw_object(name):
        """The column as its native labels — no float cast, no finiteness gate.

        Missing labels are rejected by the level validation in
        ``_categorical_codes``, which reports them in the fitted spec's terms.
        """
        if name not in raw_labels:
            raw_labels[name] = np.asarray(frame.column_array(name)).ravel()
        return raw_labels[name]

    def _margin_source(name):
        """(effective spec, x values) for a margin that grids on a spline.

        A spline-mode OrderedCategorical contributes its inner spline on the mapped
        level scores, as its refit does; every other spec contributes itself.
        """
        if name not in margin_source_cache:
            spec = model._specs[name]
            if isinstance(spec, OrderedCategorical):
                eff_spec, x = resolve_interaction_parent(spec, _raw_object(name))
                if not np.all(np.isfinite(x)):
                    raise ValueError(
                        f"screen_interactions requires finite covariates; {name!r} "
                        "maps to non-finite scores"
                    )
                margin_source_cache[name] = (eff_spec, x)
            else:
                margin_source_cache[name] = (spec, _raw_numeric(name))
        return margin_source_cache[name]

    def _margin_spec(name):
        """The spec whose geometry this margin's block is built from.

        A categorical margin is its own spec (resolving it would read labels as
        numbers); any other margin is its effective spec.
        """
        if margin_kinds[name] == "categorical":
            return model._specs[name]
        return _margin_source(name)[0]

    for name in sorted({name for pair in pairs for name in pair}, key=repr):
        # Categorical margins are read through their level labels, others through their
        # resolved source; reading all here keeps validation ahead of every statistic.
        if margin_kinds[name] == "categorical":
            _raw_object(name)
        else:
            _margin_source(name)

    distribution, link = model._distribution, model._link
    offset_inherited = False
    if offset is None:
        offset = getattr(model, "_fit_offset", None)
        offset_inherited = offset is not None
        if offset is None and getattr(model, "_fit_used_offset", False):
            raise ValueError(
                "the model was fitted with an offset but its fit state was released "
                "(retain_fit_state=False); pass offset explicitly"
            )
    if weights_inherited or offset_inherited:
        # Inherited arrays are in TRAINING row order; a reordered or edited
        # frame would silently pair them with the wrong observations.  Verify
        # against the retained fit-data guard when one is available.
        guard = getattr(model, "_fit_data_guard", None)
        if guard is not None and not guard.matches_retained_values(
            X, y, own_frame=X is getattr(model, "_fit_X_ref", None)
        ):
            raise ValueError(
                "sample_weight/offset were inherited from the fit, but X/y do not "
                "match the retained training data (reordered or modified rows); "
                "pass sample_weight and offset explicitly for non-training frames"
            )
    if offset is not None:
        offset = np.asarray(offset, dtype=np.float64)
        if offset.shape != (n_rows,):
            raise ValueError(
                f"offset must be one-dimensional with one entry per row of X; got shape "
                f"{offset.shape} for {n_rows} rows"
            )
        if not np.all(np.isfinite(offset)):
            raise ValueError("offset must contain only finite values")
    # Use the stabilized predictor directly, not link(predict(X)): for a
    # non-injective link (sqrt) the round trip flips the sign of negative-eta rows.
    # It is predict's predictor without its warning.
    from superglm.model import base

    eta = np.asarray(base.predict_eta_exact(model, X, offset, warn=False), dtype=np.float64)
    mu = np.asarray(link.inverse(eta), dtype=np.float64)
    score = working_score(y, mu, eta, weights, distribution, link)
    from superglm.solvers.working_rows import fisher_working_weights, pearson_chi2

    working_weights = fisher_working_weights(
        distribution=distribution,
        link=link,
        mu=mu,
        eta=eta,
        sample_weight=weights,
    )

    # Use the exact same family-specific residual-d.f. contract as fitting.
    # A different denominator here changes every T / phi score even though the
    # coefficients and working problem are otherwise identical.
    if phi is not None:
        phi_hat = float(phi)
    else:
        edf_mains = float(getattr(model._result, "effective_df", float("nan")))
        resolved_edf = edf_mains if np.isfinite(edf_mains) else 0.0
        denom = pearson_residual_degrees_of_freedom(
            weights,
            resolved_edf,
            weight_semantics=model_weight_semantics(model),
        )
        phi_hat = pearson_chi2(distribution=distribution, y=y, mu=mu, sample_weight=weights) / max(
            denom, 1.0
        )
        phi_hat = max(phi_hat, float(np.finfo(np.float64).tiny))

    support_cache: dict[tuple[str, bool], dict] = {}
    marginal_cache: dict[tuple[str, bool], tuple[np.ndarray, np.ndarray]] = {}
    level_cache: dict[str, tuple[np.ndarray, int]] = {}
    contrast_cache: dict[str, np.ndarray] = {}

    def _support(name, binned):
        key = (name, binned)
        if key not in support_cache:
            x = _margin_source(name)[1]
            if binned:
                x = _quantile_binned(x, screen_bins)
            uniq, codes, counts = np.unique(x, return_inverse=True, return_counts=True)
            support_cache[key] = {
                "x": x,
                "uniq": uniq,
                "codes": codes,
                "counts": counts,
                "n": len(uniq),
            }
        return support_cache[key]

    def _one_marginal(name, binned):
        """Build (menu, penalty) for a single feature; each side independently.

        Built from the margin's effective spec on the compact support (``support=``/
        ``counts=``), so a million-row feature with few distinct values never
        materializes an (n, K) basis.
        """
        key = (name, binned)
        if key not in marginal_cache:
            s = _support(name, binned)
            m = TensorInteraction._marginal_from_spec(
                _margin_spec(name),
                s["x"],
                None,
                support=s["uniq"],
                counts=s["counts"],
                # The probe's geometry as it was: SuperGLM refits no
                # interaction on a ranged parent.
                keep_ranges=False,
            )
            S = _normalize_tensor_penalty(m.penalty) if m.normalize_penalty else m.penalty
            basis = m.basis
            menu = np.asarray(basis.todense() if sp.issparse(basis) else basis, dtype=np.float64)
            marginal_cache[key] = (menu, S)
        return marginal_cache[key]

    def _levels_of(name):
        if name not in level_cache:
            level_cache[name] = _categorical_codes(model._specs[name], _raw_object(name))
        return level_cache[name]

    def _contrast_of(name):
        if name not in contrast_cache:
            contrast_cache[name] = _contrast_menu(model._specs[name])
        return contrast_cache[name]

    def _margin_support(name, binned):
        """(codes, support size) for one margin, WITHOUT building its menu.

        Allocation gates run before menu construction, so a wide factor need not
        allocate its dense (L, L-1) contrast block.
        """
        if margin_kinds[name] == "categorical":
            return _levels_of(name)
        s = _support(name, binned)
        return s["codes"], s["n"]

    def _margin(name, binned):
        """(codes, support size, menu, penalty) for one margin.

        ``binned`` applies to spline margins only.  A categorical margin's penalty is
        None: its contrast block is unpenalized.
        """
        codes, n = _margin_support(name, binned)
        if margin_kinds[name] == "categorical":
            return codes, n, _contrast_of(name), None
        menu, S = _one_marginal(name, binned)
        return codes, n, menu, S

    def _pair_penalty(S_a, S_b, k_a, k_b):
        """A factor of the pair's block penalty, or None when it has none.

        ``tensor_penalty_root(S, 0)`` factors ``kron(S, I)``, the per-level penalty a
        varying-coefficient refit applies.  The root is used rather than the assembly
        because an eigendecomposition of the assembly resolves an exactly-null
        direction only to the assembled dimension's round-off.
        """
        if S_a is None and S_b is None:
            return None
        return tensor_penalty_root(
            np.zeros((k_a, k_a)) if S_a is None else S_a,
            np.zeros((k_b, k_b)) if S_b is None else S_b,
        )

    def _within_budget(n_a, n_b, k_a, k_b):
        cells_ok = n_a * n_b <= max_cells
        inter_ok = n_a * k_b * k_b + n_b * k_a * k_a <= _INTERMEDIATE_BUDGET_FACTOR * max_cells
        return cells_ok and inter_ok

    def _numeric_margin_within_budget(k_g):
        """Budget a numeric x gridded pair BEFORE its menu is built.

        A z-moment pair allocates no cell grid; what it allocates scales with the
        gridded margin's width (one joint factor of width ``2 k_g + 3`` plus merge
        temporaries), so holding the ``(k_g + 2)^2`` overlap block to ``max_cells``
        bounds it.  Runs on the level count alone.
        """
        return (k_g + 2) ** 2 <= max_cells

    def _within_cubic_budget(k, penalized):
        """Budget the pair's SOLVE TIME, which the allocation gates do not.

        Scoring decomposes a ``(k, k)`` system, so time grows as ``k^3`` where the
        gates above grow as ``k^2``.  A penalized block is charged
        ``_PENALIZED_LADDER_COST`` times the work.  Runs on the block dimension alone;
        the constants are documented at the top of this module.
        """
        work = (_PENALIZED_LADDER_COST if penalized else 1) * int(k) ** 3
        return work <= _CUBIC_BUDGET_FACTOR * max_cells

    def _within_structured_budget(k_s, n_levels):
        """Budget the block stacks of a spline_cat pair scored through the arrow kernel.

        The pair allocates a handful of ``(n_levels, k_s + 1, k_s + 1)`` stacks; the
        persistent curvature stack and one additive suffix stack coexist during the
        profiled-trace assembly and both are charged.  The cell intermediate is
        bounded by ``_within_structured_cells`` and factor passes by
        ``_structured_evaluation_budget``.
        """
        live_stack_cells = 2 * int(n_levels) * (int(k_s) + 1) ** 2
        return live_stack_cells <= _STRUCTURED_BUDGET_FACTOR * max_cells

    def _within_structured_cells(n_a, n_levels, k_s):
        """Budget a structured pair's cell table and its curvature intermediate.

        The structured counterpart of ``_within_budget``.  The kernel forms one
        ``(n_a, k_s, k_s)`` stack of the spline menu's outer products, which scales
        with the support and which no other gate bounds.  Failing either term bins the
        spline margin first and refuses only when binning runs out.
        """
        cells_ok = int(n_a) * int(n_levels) <= max_cells
        inter_ok = int(n_a) * int(k_s) ** 2 <= _INTERMEDIATE_BUDGET_FACTOR * max_cells
        return cells_ok and inter_ok

    def _structured_evaluation_budget(n_a, k_s, n_levels):
        """Factor-pass allowance after paying for structured setup.

        Each arrow factorization scales cubically in spline width.  The work estimate
        deducts compressed-cell QR setup and per-level merges, then converts the
        remainder to a factor-pass count, reserving the worst-case search cost plus a
        final variance pass per distinct emitted lambda (three for a ladder clamped to
        one edge).  The count is neither a flop bound nor a wall-clock guarantee; see
        ``notes/research/2026-09-psst-reference-variance.md``.
        """
        return _structured_evaluation_allowance(max_cells, n_a, k_s, n_levels)

    rows = []
    from superglm.dm_builder import resolve_discrete_n_bins, should_discretize

    # Discretization is decided per SPEC (spec.discrete overrides the model flag)
    # and is lossless when a margin's cardinality fits its bin count, so approx
    # flags only a refit whose basis differs from the probe.
    model_discrete = bool(getattr(model, "_discrete", False))
    n_bins_config = getattr(model, "_n_bins", 256)

    def _refit_binned_margins(kind, feat_a, feat_b):
        """The margins whose basis the confirmatory refit would discretize.

        Mirrors the builder's gates: ti() bins only when BOTH parents discretize, a
        SplineCategorical refit when its ONE spline parent does.  Categorical margins
        never appear.
        """
        # By design: a numeric margin has no basis to bin, so numeric_cat and
        # numeric_numeric can never contribute a binned margin.
        if kind not in ("ti", "spline_cat"):
            return []
        splines = [
            name
            for name in (feat_a, feat_b)
            if margin_kinds[name] == "spline"
            and should_discretize(_margin_spec(name), model_discrete)
        ]
        if kind == "ti" and len(splines) < 2:
            return []
        return splines

    def _pair_refits_discrete(kind, feat_a, feat_b):
        # An OrderedCategorical parent refuses fit-time discretization: it already lives
        # on at most n_levels score points, so the refit sees the probe's own basis.
        if any(isinstance(model._specs[name], OrderedCategorical) for name in (feat_a, feat_b)):
            return False
        return any(
            _support(name, False)["n"]
            > resolve_discrete_n_bins(name, _margin_spec(name), n_bins_config)
            for name in _refit_binned_margins(kind, feat_a, feat_b)
        )

    for feat_a, feat_b in pairs:
        # Dispatch before any raw-column or support access, so a deferred kind
        # surfaces as its own error rather than a dtype failure downstream.
        kind = _pair_kind(margin_kinds[feat_a], margin_kinds[feat_b])
        # Set only by the spline_cat arrow path below; every other pair leaves
        # it None and is scored from its own design factor.
        structured_results = None
        if kind in ("numeric_cat", "numeric_numeric"):
            # A numeric margin enters the probe linearly: z-weighted moments over the other
            # margin's cells are exact sufficient statistics, with no grid to bin or budget
            # and no penalty, so one rung is the ladder.  Refusal (a factor too wide for its
            # own blocks) is the only degradation.
            S_root, approx = None, False
            if kind == "numeric_numeric":
                factor = numeric_numeric_factor(
                    _raw_numeric(feat_a), _raw_numeric(feat_b), score, working_weights
                )
                # Two numerics contract to a five-column design whatever the
                # supports, so no cell table stands behind them.
                n_cells = 1
            else:
                num_name, cat_name = (
                    (feat_a, feat_b) if margin_kinds[feat_a] == "numeric" else (feat_b, feat_a)
                )
                codes_g, n_g = _margin_support(cat_name, False)
                # The factor's levels ARE the cells: the numeric side contributes
                # z-moments within them rather than cells of its own.
                n_cells = n_g
                # Exact, not an estimate: a factor margin's contrast menu is
                # (L, L-1) wide, so the gate needs no post-menu recheck.
                k_g = _marginal_width_estimate(_margin_spec(cat_name))
                if not (
                    _numeric_margin_within_budget(k_g)
                    # ... and the (L-1)-wide blocks must be SOLVABLE in
                    # bounded time, not merely allocatable: the pair's own
                    # factorizations are cubic in the same width.
                    and _within_cubic_budget(k_g, False)
                ):
                    rows.append(
                        (feat_a, feat_b, kind, np.nan, np.nan, np.nan, np.nan, n_cells, approx)
                    )
                    continue
                _, _, menu_g, _ = _margin(cat_name, False)
                factor = numeric_cat_factor(
                    codes_g, n_g, menu_g, _raw_numeric(num_name), score, working_weights
                )
        else:
            # Assemble with the categorical margin LAST so a mixed pair's penalty is
            # kron(S_spline, I), the varying-coefficient layout.  The statistic is invariant
            # to the column permutation; reported feature_a/feature_b keep the caller's order.
            left, right = feat_a, feat_b
            if margin_kinds[left] == "categorical" and margin_kinds[right] != "categorical":
                left, right = right, left
            k_l = _marginal_width_estimate(_margin_spec(left))
            k_r = _marginal_width_estimate(_margin_spec(right))
            bin_flag = {left: False, right: False}
            margins = None
            structured = False
            arrow_budget = 0
            # The dense path gets first refusal, so nothing it can already
            # score EXACTLY changes.  Cleared when it runs out of exact moves,
            # which hands a spline_cat pair to the arrow kernel below.
            allow_dense = True
            # Set when the handoff was taken speculatively on a width estimate biased low:
            # if the true width falls outside the arrow budgets, the dense path regains its
            # binning fallback instead of the pair becoming a NaN row.
            arrow_lookahead = False
            # A failed handoff cannot repeat on the same support. Keep
            # numerical certification refusals latched for the whole pair.
            arrow_refused = False
            # Budget refusals are local to the support whose work was too
            # expensive; binning can make another attempt affordable.
            arrow_budget_refused = False
            while True:
                codes_l, n_l = _margin_support(left, bin_flag[left])
                codes_r, n_r = _margin_support(right, bin_flag[right])
                # Both gates below bound a DENSE (k_l*k_r, k_l*k_r) block: its
                # allocation, and the cubic solve it feeds.  Binning can shrink
                # neither, since neither depends on the support.
                dense_ok = allow_dense and (
                    (k_l * k_r) ** 2 <= _INTERMEDIATE_BUDGET_FACTOR * max_cells
                    and _within_cubic_budget(k_l * k_r, kind in ("ti", "spline_cat"))
                )
                # A spline x categorical pair never has to form the dense block: grouped by
                # level its bordered system is an arrow matrix, factorized in time and memory
                # linear in the level count.  A pair the dense gates refuse is retried against
                # the structured gate.
                structured = kind == "spline_cat" and not dense_ok
                structured_setup_ok = True
                if not (dense_ok or structured):
                    break
                if structured:
                    # The block stacks do not depend on support, but the profiled-trace setup does
                    # scale with n_l, so an unaffordable exact support must reach the spline-binning
                    # fallback below.
                    arrow_budget = _structured_evaluation_budget(n_l, k_l, n_r)
                    if not _within_structured_budget(k_l, n_r):
                        if arrow_lookahead:
                            # Speculation did not survive the true width; put
                            # the dense path back on the track it was on.
                            allow_dense, arrow_lookahead = True, False
                            continue
                        break
                    structured_setup_ok = arrow_budget >= 2
                    if not structured_setup_ok and arrow_lookahead:
                        allow_dense, arrow_lookahead = True, False
                        continue
                # Both paths build the (n_l, n_r) cell tables and a curvature
                # intermediate that scales with the support -- transposed
                # between them, so each has its own second term.
                fits = (
                    structured_setup_ok and _within_structured_cells(n_l, n_r, k_l)
                    if structured
                    else _within_budget(n_l, n_r, k_l, k_r)
                )
                if fits:
                    _, _, menu_l, S_l = _margin(left, bin_flag[left])
                    if structured:
                        # The contrast menu is one-hot, so the kernel takes the
                        # rows it indicates rather than the menu itself — the
                        # menu is the one allocation that is quadratic in L.
                        level_rows = _contrast_rows(model._specs[right])
                        if (menu_l.shape[1], level_rows.size) != (k_l, k_r):
                            k_l, k_r = menu_l.shape[1], int(level_rows.size)
                            continue
                        # The ladder runs here because whether the arrow path can score a pair is a
                        # routing question: its cost depends on the penalty's null space, so no gate can
                        # predict it.  A refused speculative handoff hands the dense track back.
                        try:
                            # Two endpoints and a final variance pass are a lower bound for any penalized
                            # ladder; reject before allocating cell tables.
                            if arrow_budget < 3 and np.any(S_l):
                                raise _StructuredBudgetExceededError
                            S_cell, W_cell = pair_cell_moments(
                                codes_l,
                                codes_r,
                                n_l,
                                n_r,
                                score,
                                working_weights,
                                max_cells=max_cells,
                            )
                            structured_results = structured_ladder(
                                spline_cat_moments(menu_l, S_l, S_cell, W_cell, level_rows),
                                budgets=budgets,
                                max_evaluations=arrow_budget,
                                raise_on_budget=True,
                            )
                        except _StructuredBudgetExceededError:
                            # Binning can free work for search and variance passes.  A speculative refusal
                            # must latch, or the unchanged dense state retries the same handoff.
                            if arrow_lookahead:
                                allow_dense, arrow_lookahead = True, False
                                arrow_budget_refused = True
                                continue
                        else:
                            if structured_results is None and arrow_lookahead:
                                allow_dense, arrow_lookahead = True, False
                                arrow_refused = True
                                continue
                            margins = ((menu_l, S_l), (level_rows, None))
                            break
                    else:
                        _, _, menu_r, S_r = _margin(right, bin_flag[right])
                        if (menu_l.shape[1], menu_r.shape[1]) != (k_l, k_r):
                            # Re-run the gates with the built menus' dimensions.
                            k_l, k_r = menu_l.shape[1], menu_r.shape[1]
                            continue
                        margins = ((menu_l, S_l), (menu_r, S_r))
                        break
                # bin the largest not-yet-binned margin that binning can shrink;
                # a categorical margin is never binnable, whatever its level count
                binnable = [
                    (n, name)
                    for n, name in ((n_l, left), (n_r, right))
                    if margin_kinds[name] == "spline" and not bin_flag[name] and n > screen_bins
                ]
                try:
                    # A size tie breaks on the labels' own order, as on 0.40.
                    binnable = sorted(binnable, reverse=True)
                except TypeError:
                    # Labels that do not order (0 and "z") break it on their text.
                    binnable = sorted(
                        binnable, key=lambda item: (item[0], str(item[1])), reverse=True
                    )
                if not binnable:
                    if arrow_lookahead:
                        # The speculative handoff has run out of support fallback too.  Hand the dense
                        # track back rather than trading a scored (approximate) row for a NaN; the same
                        # restore the width recheck above performs.
                        allow_dense, arrow_lookahead = True, False
                        continue
                    # The dense path has no moves left.  Try the arrow kernel for a spline_cat pair
                    # before giving up: the dense-track width estimate is biased low by design.
                    if (
                        allow_dense
                        and not structured
                        and kind == "spline_cat"
                        and not (arrow_refused or arrow_budget_refused)
                    ):
                        allow_dense = False
                        continue
                    break
                if allow_dense and not structured and kind == "spline_cat":
                    # About to compress the spline margin, after which the dense path can no longer
                    # score the pair exactly.  The arrow path may still: its intermediate is the
                    # transpose of the one that failed.  Prefer an exact score to an approximate one.
                    # `left` is the spline margin for spline_cat, by the swap above.
                    if (
                        not (arrow_refused or arrow_budget_refused)
                        and _within_structured_budget(k_l, n_r)
                        and _structured_evaluation_budget(n_l, k_l, n_r) >= 2
                        and _within_structured_cells(n_l, n_r, k_l)
                    ):
                        allow_dense, arrow_lookahead = False, True
                        continue
                bin_flag[binnable[0][1]] = True
                # The refusal applies only to the unchanged support. Binning
                # frees structured work even when the dense intermediate
                # still does not fit, so allow a fresh handoff on this state.
                arrow_budget_refused = False
            approx = (
                bin_flag[left] or bin_flag[right] or _pair_refits_discrete(kind, feat_a, feat_b)
            )
            n_cells = n_l * n_r
            if margins is None:
                rows.append((feat_a, feat_b, kind, np.nan, np.nan, np.nan, np.nan, n_cells, approx))
                continue

            (menu_l, S_l), (menu_r, S_r) = margins
            if structured:
                if structured_results is None:
                    # Numerical certification refused every rung. Budget
                    # refusals already tried binning inside the routing loop.
                    # A speculative refusal restores the dense track above.
                    rows.append(
                        (feat_a, feat_b, kind, np.nan, np.nan, np.nan, np.nan, n_cells, approx)
                    )
                    continue
            else:
                S_cell, W_cell = pair_cell_moments(
                    codes_l, codes_r, n_l, n_r, score, working_weights, max_cells=max_cells
                )
                factor = pair_design_factor(menu_l, menu_r, S_cell, W_cell)
                S_root = _pair_penalty(S_l, S_r, menu_l.shape[1], menu_r.shape[1])
        # An unpenalized block has no bandwidth to scan: every rung returns the same
        # rank, statistic and lambda0=0, so one rung is the ladder.  A penalty whose
        # directions are all round-off arrives as an empty root.
        penalized = structured_results is None and S_root is not None and S_root.size > 0
        # The whole ladder shares ONE decomposition, clamped rungs included.  The pencil
        # that turns edf(lambda) and T(lambda) into closed forms depends on neither
        # lambda nor edf0, so every rung after the first costs O(k).  Recomputing rungs
        # that clamp to an identical triple cannot displace the incumbent, since the
        # comparison below is strict.
        results = structured_results
        if results is None:
            results = penalized_score_statistic_ladder(
                factor,
                S_root,
                budgets=tuple(float(b) for b in (budgets if penalized else budgets[:1])),
            )
        best_z, best = -np.inf, None
        for result in results:
            if not result.edf0 > 0.0 or not result.reference_variance > 0.0:
                # A rung that resolved no direction has no test to run, and the normalization
                # would report z = inf and sort an uninformative pair to the top.  This is a
                # pure arithmetic guard (rank 0), not a weak-identification threshold.
                continue
            statistic = result.statistic / phi_hat
            z = (statistic - result.edf0) / np.sqrt(result.reference_variance)
            if z > best_z:
                best_z, best = z, (statistic, result.edf0, result.lambda0)
        if best is None:
            rows.append((feat_a, feat_b, kind, np.nan, np.nan, np.nan, np.nan, n_cells, approx))
        else:
            rows.append((feat_a, feat_b, kind, best[0], best_z, best[1], best[2], n_cells, approx))

    table = pd.DataFrame(rows, columns=_RESULT_COLUMNS)
    table = table.sort_values("z", ascending=False, ignore_index=True)
    table.attrs["phi"] = phi_hat
    table.attrs["deferred_features"] = deferred_features
    return table
