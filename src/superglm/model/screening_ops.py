"""PSST interaction screening over a fitted mains model.

PSST — Penalized Smooth Score Test — ranks every candidate pair by how much
of the fitted model's leftover working signal the interaction block the pair
would actually refit as could absorb at a fixed screening complexity.  One
fused O(n) cell pass per pair, no refits; the confirmatory ``fit_reml``
refit of the top-ranked pairs is the gate.  Ranking-only: the statistic is
not a calibrated p-value and must not be reported as one.

The sweep is not spline-only.  The ``kind`` column names the interaction
class each pair would refit as — ``ti`` (spline x spline), ``spline_cat``,
``cat_cat``, ``numeric_cat`` and ``numeric_numeric``.  The gridded kinds run
through the same cell kernels on a per-margin ``(codes, support size, menu,
penalty)`` description: a categorical margin is a gridded margin over its
fitted levels whose menu is the (L, L-1) treatment-contrast block and whose
penalty is absent, so ``kron`` of two such menus reproduces the cross-product
indicator columns and ``kron`` of a spline menu with one reproduces the
per-level curve blocks, column for column.  A NUMERIC margin never grids: it
enters its probe linearly (a per-level slope, or a product of two numerics),
so z-weighted moments accumulated over the other margin's cells are the exact
sufficient statistics — see ``screening/_numeric_margin``.  Such a pair is
therefore exact whenever it is computed at all: it has no binning fallback,
so a factor margin too wide for the pair's blocks is REFUSED with a NaN row
rather than approximated.
A spline-mode ``OrderedCategorical`` margin rides the spline arm on its
MAPPED level scores — the geometry its own refit builds — so its pairs are
``ti`` and ``spline_cat`` like any other spline margin's, gridded on at most
``n_levels`` support points.
``z`` normalizes each kind against its own noise floor, so a single sorted
table ranks them together — but not on equal terms, and not simply by df.
The measured null maxima span 3.98 to 7.53 across kinds; the heaviest tails
sit at low probe df, yet "neither kind is monotone in df", and the headline
maxima read as "Gaussian-driven rather than as something every family
reproduces".  A maximum also grows with the number of draws, so a wide sweep
draws more null rows than a narrow one.  Compare like with like before
spending a refit, against the measured floors in the screening guide.
A pair with no penalty anywhere
in its block has
no bandwidth to scan and is evaluated at a single rung: ``edf0`` then reports
the block's achieved rank and ``lambda0`` is 0.

``max_cells`` bounds two different resources.  It is an ALLOCATION ceiling
for the cell tables, the curvature intermediates and a numeric-margin pair's
blocks; through ``_within_cubic_budget`` it is also a TIME ceiling, because
the probe's block dimension ``k`` enters every rung as a ``(k, k)``
factorization or pseudo-inverse.  Per-pair time therefore grows as ``k^3``
where every allocation here grows as ``k^2``, and for the gridded kinds the
pseudo-inverse branch is the routine path rather than the exception — one
empty ``cat_cat`` cell or one singleton factor level makes a probe column
collinear with the overlap span and the Cholesky falls back.  A block too
wide to SOLVE inside the budget is refused with a NaN row exactly as an
unaffordable allocation is, and refused immediately: binning cannot shrink a
basis dimension, so no fallback is attempted first.

The statistic is reported on the ``T / phi`` scale, with ``phi`` the mains
fit's Pearson dispersion estimate: under the null ``E[T] = phi * edf0``, so
without this scaling the ``edf0`` noise floor is only honest for
unit-dispersion families and a dispersed Gaussian null would swamp the scan.
The Pearson denominator follows the same family-specific weight contract as
the fitted model.  Non-Tweedie families use case/frequency weights and
therefore ``sum(w) - edf``; Tweedie uses EDM prior weights and therefore the
positive-observation count minus ``edf``.  Keeping the screen and published
fit on one scale is essential because ``T / phi`` drives the ranking.

Per-feature work (unique support, codes, marginal basis menus, marginal
penalties — level codes and the contrast menu for a categorical margin, the
mapped level scores for an ordered-categorical one) is
cached across the sweep, so an all-pairs screen over P screened features
builds each feature's marginal exactly once.  The caches trade
memory for time and live for the whole sweep: roughly the raw covariates
plus codes plus menus per screened feature (plus binned variants where the
fallback fires, and the mapped scores beside the labels for an
ordered-categorical margin), instead of per-pair transients.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from superglm._frame import as_eager_frame
from superglm.distributions import _VARIANCE_FLOOR, Tweedie, validate_response
from superglm.features.categorical import (
    Categorical,
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
    numeric_numeric_moments,
    numeric_pair_moments,
    pair_cell_moments,
    pair_score_curvature,
    penalized_score_statistic_ladder,
    working_score,
)
from superglm.screening._overlap import pair_overlap_moments, tensor_penalty
from superglm.screening._structured import spline_cat_moments, structured_ladder
from superglm.solvers.dispersion import pearson_residual_degrees_of_freedom

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

# The V-assembly einsum carries (n_a, k_b, k_b) and (n_b, k_a, k_a)
# intermediates that max_cells alone does not bound (a lopsided 1e6 x 5 pair
# passes the cell budget while the intermediate is 40x larger).  Budget them
# against a small multiple of max_cells; the marginal dimension is estimated
# from the parent spec's k, which is a guard-grade bound, not an exact count.
_INTERMEDIATE_BUDGET_FACTOR = 4

# Every budget above bounds an ALLOCATION.  Per-pair TIME is cubic in the
# probe block's dimension k, because each rung factorizes or pseudo-inverts a
# (k, k) system, and the allocation budgets alone admit blocks whose solve
# costs minutes: at the default they let a cat_cat pair reach k = 4472.
# Measured end to end through screen_interactions on the reference box with a
# single BLAS thread, per k^3 of block dimension:
#   2.9e-10 s  unpenalized block, pseudo-inverse path (cat_cat with an empty
#              cell or a singleton level -- the routine case: 24 s and 1.3 GB
#              at two 67-level factors, k = 4290)
#   2.0e-10 s  unpenalized block, Cholesky path (numeric_cat, full rank:
#              2.1 s at the widest factor the allocation gate admits)
#   2.7e-10 s  penalized block, WHOLE four-rung ladder
# The two UNPENALIZED figures, and the 1.6e-10 s/k^3 (0.81 s at k = 1709)
# below, are the cost before issue #199 replaced that rung's edf -- a Cholesky
# trace, which reports k rather than the rank whenever cho_factor accepts a
# barely definite block -- with a rank COUNT.  Counting is an eigenvalue
# decomposition where the trace was a k-right-hand-side solve, so it is not
# free.  Measured as an interleaved A/B in ONE process, single BLAS thread, so
# that load drift cannot be mistaken for the change: 1.056x at the numeric_cat
# corner the budget below actually admits (k = 1709, one singleton level:
# 1.037 s -> 1.095 s, and the reported edf0 1709 -> 1708, which is the point)
# and 1.209x on the pseudo-inverse path at two 67-level factors (k = 4356, one
# empty cell: 14.0 s -> 16.9 s, 1.48 GB either way).  Carry the figures above
# forward by those multipliers; the budget ceiling then lands near 1.1 s per
# unpenalized pair, still inside the ~1.5 s target, so no constant here moves.
# Holding the worst path of each class to ~1.5 s per pair gives k^3 <=
# _CUBIC_BUDGET_FACTOR * max_cells for an unpenalized block, and the same
# budget against _PENALIZED_LADDER_COST times the work for a penalized one.
#
# That multiplier used to be 16.  A penalized ladder re-solved the (k, k)
# system ~27 times PER RUNG to bisect for its edf target -- 109 solves per
# pair, 5.0e-09 s/k^3, 29 s at k = 1849 -- and since whether a rung bisects is
# not knowable before the solve, the budget had to assume it did.  The ladder
# now brackets once for the whole ladder and, only when some rung genuinely
# has to search, shares ONE simultaneous diagonalization of the pencil
# (V_eff, S) across every rung that does, which makes edf(lambda) and
# T(lambda) closed forms and reduces the search to O(k) arithmetic on them.
# Measured after that change, at k = 1709, one BLAS thread: 1.6e-10 s/k^3
# unpenalized against 2.7e-10 for the entire four-rung penalized ladder --
# a ratio of 1.65, not 25.  End to end on a bisecting 28x28-knot ti the same
# change is 4.20 s -> 0.54 s.  The multiplier is set to 2 rather than 1.65 to
# keep headroom for the pseudo-inverse branch, which either path can take.
#
# At the default max_cells this admits k <= 1709 unpenalized and k <= 1357
# penalized, measured there at 0.81 s and 0.67 s -- both inside the 1.5 s
# target the old constants were chosen against.  Raising max_cells lifts
# both, as it lifts every other budget here.  The remaining ceiling is an
# artifact of densifying a block that is structurally block-diagonal; see
# issue #188.
_CUBIC_BUDGET_FACTOR = 1000
_PENALIZED_LADDER_COST = 2

# Every constant above budgets a DENSE block.  A spline x categorical pair
# refused by them is retried through the arrow kernel, whose cost is linear in
# the level count rather than cubic; see screening/_structured.py.  It needs
# the same two KINDS of budget the dense path needs and for the same reasons:
# _within_structured_budget and _within_structured_cells bound allocations,
# which grow as k_s^2, and _structured_evaluation_budget bounds solve time,
# which grows as k_s^3 -- one batched eigendecomposition of (k_s + 1) blocks
# per level per evaluation, and a ladder that has to bisect runs tens of them.
# The cubic factor retains the original arrow-factorization calibration.  The
# stable profiled-trace setup introduced for issue #204 is now deducted from
# that same work ceiling before any endpoint evaluations are admitted.
# Two level-sized stacks coexist while the stable profiled trace is assembled:
# the pair's persistent curvature blocks and one additive suffix stack.  The
# factor is written as two because _within_structured_budget charges both
# explicitly; the resulting admission ceiling is the same one-stack ceiling
# used before issue #204 rather than an unaccounted doubling.
_STRUCTURED_BUDGET_FACTOR = 2
_STRUCTURED_CUBIC_BUDGET_FACTOR = 50


def _structured_evaluation_allowance(max_cells, n_a, k_s, n_levels):
    """Arrow evaluations left after the stable profiled-trace setup."""
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
        # A term with specials= is refused HERE, ahead of the basis test and
        # before any column is read.  A special is a free level with no
        # position on the spline axis, so ``resolve_interaction_parent``
        # refuses it -- and that resolver runs inside the eager pre-read
        # below, which would abort the WHOLE sweep on the first specials term
        # rather than skipping it.  Screening one needs composite margins.
        if spec.has_specials:
            return None
        # A spline-mode OC is a spline through the level values, so it screens
        # (and refits) exactly like one; step mode has no interaction target.
        if spec.basis == "spline" and spec._spline is not None:
            return "spline"
        return None
    if isinstance(spec, Categorical):
        # KEPT as a guard, not a live case: Categorical.build raises below two
        # levels, so a FITTED spec always clears this and the None branch is
        # unreachable from screen_interactions.  Left in place because the
        # alternative is a (1, 0) contrast menu reaching the cell kernels.
        return "categorical" if len(spec._levels) >= 2 else None
    if isinstance(spec, Numeric):
        return "numeric"
    return None


def _deferral_reason(spec) -> str:
    """Why a fitted main effect has no screenable margin.

    Called only for names ``_margin_kind`` refused, so every branch is a
    deferral rather than an error, and the reason is REPORTED --- on
    ``table.attrs["deferred_features"]`` and in the candidates error --- rather
    than dropped on the floor.  Polynomial and step-mode OrderedCategorical
    were silently skipped before this existed, which is the same defect.
    """
    if isinstance(spec, OrderedCategorical):
        if spec.has_specials:
            return (
                "OrderedCategorical with specials= is deferred: a special is a free "
                "level with no position on the spline axis, so the margin has no "
                "score to grid on; screening the pair needs composite margins"
            )
        # Mirror ``_margin_kind``'s disjunction rather than assuming its
        # complement: it refuses on ``basis != "spline" OR _spline is None``,
        # so naming every survivor "step-mode" would mislabel a spline-mode
        # term that reached here without an inner spline.
        if spec.basis != "spline":
            return (
                "step-mode OrderedCategorical is deferred: the one-hot geometry, "
                "removed in 0.24.0, has no marginal smooth to cross with"
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
        # A pair the model already fits as an interaction is not a candidate:
        # the screen profiles only the parent mains, so it would re-surface
        # the fitted interaction and the confirmation workflow would then
        # fail with "interaction already added".
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
            # A name the model DID fit but cannot screen (Polynomial, step-mode
            # or specials OrderedCategorical) is deferred, not a typo; listing
            # the screenable features would send the caller hunting for a
            # misspelling that isn't there.  The reason quoted here is the same
            # string the result table reports.
            deferred_names = sorted(name for name in pair if name in deferred_features)
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
                f"{sorted(margin_kinds)})"
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
    """Dense 0-based codes over ALL fitted levels, in ``spec._levels`` order.

    Applies the same grouping collapse and unseen-level validation the fitted
    spec's ``transform`` applies, so the screen sees exactly the mains' level
    geometry — including the BASE level, which the main effect absorbs into
    the intercept but the screen needs a grid row for.
    """
    x = _resolve_categorical_labels(
        x_raw,
        spec._grouping,
        known_levels=set(spec._levels),
    )
    codes = pd.Categorical(x, categories=spec._levels).codes.astype(np.intp)
    if codes.size and codes.min() < 0:
        # Reachable through a grouping whose collapsed label was absent from
        # the training data: validation passes on the original level, but the
        # group it maps to was never fitted and has no grid row.
        raise ValueError(
            "screen_interactions found categorical values outside the fitted "
            f"level set {list(spec._levels)}"
        )
    return codes, len(spec._levels)


def _contrast_menu(spec) -> np.ndarray:
    """(L, L-1) treatment-contrast menu; the base level's row is all zeros.

    Column ``j`` indicates ``spec._non_base[j]``, matching the main effect's
    column order, so the interaction block this menu generates is exactly the
    one the confirmatory refit would build.
    """
    position = {lev: i for i, lev in enumerate(spec._levels)}
    menu = np.zeros((len(position), len(position) - 1), dtype=np.float64)
    for j, lev in enumerate(spec._non_base):
        menu[position[lev], j] = 1.0
    return menu


def _contrast_rows(spec) -> np.ndarray:
    """The one row each contrast column indicates — the menu, without the menu.

    ``_contrast_menu`` is one-hot by construction, so multiplying by it is
    selecting ``L - 1`` columns of whatever it multiplies.  The structured
    kernel takes these indices instead of the menu, which is the difference
    between ``L - 1`` integers and an ``(L, L - 1)`` block of doubles: 20 GB
    at fifty thousand levels, holding 49,999 nonzeros.
    """
    position = {lev: i for i, lev in enumerate(spec._levels)}
    return np.fromiter(
        (position[lev] for lev in spec._non_base), dtype=np.intp, count=len(spec._non_base)
    )


def _marginal_width_estimate(spec) -> int:
    """Guard-grade marginal width from the parent spline's geometry.

    Takes the margin's EFFECTIVE spec — the caller resolves an
    OrderedCategorical margin to its inner spline first, because the wrapper's
    own ``n_knots`` can predate the level-count clamp and would over-estimate.

    Deliberately biased LOW: an under-estimate self-heals (menus get built
    and the authoritative post-menu recheck re-runs the gates with true
    dimensions), while an over-estimate would bin or skip a pair that fits
    the budget with no correction possible.  ``n_knots`` floors the centered
    marginal width of every built-in kind, including degree-0 ps/bs
    (``n_knots + degree``) and cr via the CardinalCR substitution
    (``n_knots + 1``).  Categorical and numeric widths are exact, not
    estimates: a contrast menu is (L - 1) wide and a numeric margin is 1.
    """
    if isinstance(spec, Categorical):
        return max(len(spec._levels) - 1, 1)
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

    Each pair is screened as the interaction class it would refit as, named
    in the ``kind`` column: ``ti`` for spline x spline, ``spline_cat`` for a
    spline crossed with a factor, ``cat_cat`` for two factors, ``numeric_cat``
    for a per-level numeric slope and ``numeric_numeric`` for a product of two
    numerics.  A spline-mode ``OrderedCategorical`` margin screens as a spline
    on its mapped level scores, so its pairs carry the spline kinds.
    ``z`` normalizes each kind against its own noise floor, so one
    sorted table ranks them together — but not on equal terms, and not simply
    by df: the measured null maxima span 3.98 to 7.53 across kinds.  The
    heaviest tails sit at low probe df, yet "neither kind is monotone in df",
    and the headline maxima read as "Gaussian-driven rather than as something
    every family reproduces".  Compare like with like before spending a
    refit — see the measured floors in the screening guide.
    A kind whose block carries no penalty
    (``cat_cat``, ``numeric_cat``, ``numeric_numeric``) has no bandwidth to
    scan and is evaluated at a single rung — ``edf0`` then reports the block's
    achieved rank and ``lambda0`` is 0, so the ``edf0`` argument does not
    apply to it.

    ``edf0`` is the probe bandwidth: a smooth surface is detected best by a
    small budget, a high-frequency one only by a budget at least as complex
    as its shape (measured: a sin x sin signal is invisible at edf0<=4).  The
    default is therefore a LADDER — each pair is evaluated at every budget,
    each T is normalized against its own noise floor,
    ``z = (T - edf0) / sqrt(2 * edf0)``, and the pair is ranked by its best
    normalized score, a scan statistic over bandwidths.  Pass a single float
    to probe one bandwidth.  The expensive per-pair work (cells, menus,
    profiling) happens once; the ladder re-solves a small system per rung.

    ``offset`` and ``sample_weight`` both default to the values the model
    was fitted with (weights only when the fit's were non-unit), so the
    screen linearizes at the fitted likelihood.  Inherited arrays are in
    training row order, so inheriting requires ``X``/``y`` to BE the
    retained training data — screening a holdout, subsample, or reordered
    frame must pass ``sample_weight`` (and ``offset``) explicitly.  ``T``
    is scaled by the fit's
    Pearson dispersion estimate before normalization (see module docstring);
    the value used is attached as ``table.attrs["phi"]`` and can be
    overridden with ``phi=`` for a deliberately different calibration study.

    Fitted mains with no screenable margin — ``Polynomial``, ``RandomEffect``,
    step-mode ``OrderedCategorical`` and any ``OrderedCategorical`` carrying
    ``specials=`` — are excluded from the sweep and reported in
    ``table.attrs["deferred_features"]``, a ``{feature: reason}`` mapping that
    is empty when everything fitted was screened.  Naming one of them in
    ``candidates`` raises with the same reason.

    Returns a frame sorted by ``z`` (descending) with one row per screened
    pair: ``feature_a, feature_b, kind, statistic, z, edf0, lambda0, n_cells,
    approx``, where ``kind`` names the interaction class the pair would refit
    as and ``statistic``/``edf0``/``lambda0`` describe the winning rung.
    ``statistic`` is therefore NOT comparable across rows (rungs
    differ) — rank by ``z``.  At a clamped rung ``edf0`` holds the achieved
    value and ``lambda0`` is a bracket edge, not an interpretable smoothing
    parameter.

    Pairs whose support exceeds the cell or intermediate budgets fall back
    to quantile binning, largest margin first: a margin with more than
    ``screen_bins`` unique values is compressed to ``screen_bins``
    empirical-quantile bins (basis evaluated at within-bin means) and the
    row is flagged ``approx=True``.  Pairs within budget are always computed
    exactly — the fallback never touches them — though ``approx`` can still
    be True for such a pair when its refit would discretize lossily.
    A pair still over budget after binning is skipped with a NaN row
    (``n_cells`` reports the grid that was attempted and ``approx`` whether
    binning was applied), as is a pair whose statistic degenerates.  A pair
    whose tensor curvature block ``(k_a*k_b)^2`` alone exceeds the budget is
    skipped immediately — binning cannot shrink basis dimensions, so such
    rows may report the raw grid with no binning attempted.

    ``max_cells`` bounds allocation AND time.  The probe block's dimension
    ``k`` enters every rung as a ``(k, k)`` factorization or pseudo-inverse,
    so per-pair time grows as ``k^3`` where the allocations grow as ``k^2``
    — and for the gridded kinds the pseudo-inverse is the routine branch, not
    the exception (one empty ``cat_cat`` cell or one singleton level makes a
    probe column collinear with the overlap span).  ``_within_cubic_budget``
    therefore refuses a pair whose block is too wide to solve inside the
    budget: ``k^3 <= 1000 * max_cells`` for an unpenalized block, and the
    same budget against twice the work for a penalized one whose ladder can
    bisect.  At the default that admits ``k <= 1709`` and ``k <= 1357``
    respectively; the times measured at those ceilings are in this module's
    docstring, and raising ``max_cells`` lifts both.  Binning cannot shrink a
    basis dimension, so the refusal is immediate and no fallback is
    attempted.  A ``spline_cat`` pair refused here is retried through the
    arrow kernel, which has no block-dimension ceiling but budgets of its
    own: the level blocks must fit ``max_cells``, the spline menu's outer
    products must fit the same intermediate budget the dense path uses (and
    are quantile-binned first if they do not), and the ladder must be able to
    afford its arrow factorizations — a pair whose rungs have to bisect costs
    tens of them and is refused with a NaN row when that does not fit.
    ``approx`` is also True for any pair whose confirmatory refit would
    discretize LOSSILY, applying the gate that refit itself uses — which
    differs by kind.  A ``ti`` refit bins its marginal supports only when
    BOTH parents resolve to fit-time discretization (per-spec ``discrete``
    overriding the model flag); a ``spline_cat`` refit bins its spline margin
    whenever that ONE parent does.  Either way the row is flagged only when
    some margin the refit would bin has a cardinality exceeding its resolved
    bin count: lossless binning returns the exact unique support, so those
    refits match the probe basis and stay ``approx=False``.  A categorical
    margin never bins and never discretizes — its support is the fitted level
    set — so it never contributes to the flag, and a ``cat_cat`` row is
    always ``approx=False``.  A numeric margin never contributes either: it
    enters its probe linearly, with no grid to build, no support to compress
    and no basis a refit could discretize, so ``numeric_cat`` and
    ``numeric_numeric`` rows never reach the binning fallback and are always
    ``approx=False``.  Every such row that carries a statistic is exact; the
    only degradation available to these kinds is REFUSAL, since there is
    nothing to approximate.  A ``numeric_cat`` pair is refused with a NaN row
    when the factor is too wide for the pair's blocks to fit ``max_cells``
    (the ``(L+1)``-wide overlap curvature is the largest of them, admitting
    factors up to 2235 levels at the default) or too wide for them to be
    SOLVED inside it (``(L-1)^3 <= 1000 * max_cells``, which binds first at
    the default and admits 1710 levels); raising ``max_cells``
    lifts the refusal and the pair is then computed exactly.  A
    ``numeric_numeric`` pair contracts to 3x3 blocks and is never refused.
    Their ``n_cells`` counts the OTHER margin's cells alone — the factor's
    fitted levels, or 1 when both margins are numeric.
    """
    if getattr(model, "_result", None) is None:
        raise RuntimeError("screen_interactions requires a fitted model; call fit_reml first")
    from superglm.features.interaction import TensorInteraction, _normalize_tensor_penalty

    frame = as_eager_frame(X)
    n_rows = len(frame)
    y = np.asarray(y, dtype=np.float64)
    weights_inherited = False
    if sample_weight is None:
        # Only a genuinely non-unit fitted weight vector is worth inheriting:
        # a unit-weight fit screens any frame without arguments (ones cannot
        # mispair rows), and inheriting them anyway would needlessly pin X/y
        # to the training data via the fit-data guard below.  Non-unitness is
        # derived from the STORED array, not the _fit_used_weights stamp: the
        # editor may rewrite _fit_weights without touching the stamp, and the
        # array is the ground truth of the published fit state.
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
    if isinstance(model._distribution, Tweedie) and (
        not np.all(np.isfinite(weights)) or np.any(weights <= 0.0)
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
        # Only spline margins carry a double-penalty flag; an OC margin's
        # lives on its inner spline.  Unwrapped rather than resolved through
        # ``_margin_spec``: this gate runs before any column is read, so that
        # an unbuildable parent is named upfront rather than after the sweep
        # has paid for (and possibly failed on) the covariates.
        spec = model._specs[name]
        if isinstance(spec, OrderedCategorical):
            spec = spec._spline
        return getattr(spec, "select", False)

    selected = sorted({name for pair in pairs for name in pair if _select_flag(name)})
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

        A spline-mode OrderedCategorical margin contributes its INNER spline
        on the mapped level scores — the same resolution the confirmatory
        refit applies, with the same grouping collapse and level validation —
        so its label column is read as labels and never float-cast.  Every
        other spec contributes itself on its own numeric column.
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

        A categorical margin is its own spec — its block is the contrast menu
        over the fitted levels, and resolving it would read the label column
        as numbers.  Any other margin is its EFFECTIVE spec, which for an
        OrderedCategorical is the inner spline that carries the menu, the
        marginal width and the penalty.
        """
        if margin_kinds[name] == "categorical":
            return model._specs[name]
        return _margin_source(name)[0]

    for name in sorted({name for pair in pairs for name in pair}):
        # A categorical margin is read through its level labels; every other
        # margin through its resolved source, which maps an OrderedCategorical's
        # labels to scores rather than casting them.  Reading them all here
        # keeps input validation ahead of every statistic.
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
        if guard is not None and not guard.matches_retained_values(X, y):
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
    # The stabilized predictor directly, NOT link(predict(X)): for a
    # non-injective link (sqrt) the round trip maps eta to |eta| and flips
    # the sign of every negative-eta row's score.
    eta = np.asarray(model._predict_eta_exact(X, offset), dtype=np.float64)
    mu = np.asarray(link.inverse(eta), dtype=np.float64)
    score = working_score(y, mu, eta, weights, distribution, link)
    dmu_deta = link.deriv_inverse(eta)
    var_mu = np.maximum(distribution.variance(mu), _VARIANCE_FLOOR)
    working_weights = weights * dmu_deta**2 / var_mu

    # Use the exact same family-specific residual-d.f. contract as fitting.
    # A different denominator here changes every T / phi score even though the
    # coefficients and working problem are otherwise identical.
    if phi is not None:
        phi_hat = float(phi)
    else:
        edf_mains = float(getattr(model._result, "effective_df", float("nan")))
        resolved_edf = edf_mains if np.isfinite(edf_mains) else 0.0
        denom = pearson_residual_degrees_of_freedom(
            distribution,
            weights,
            resolved_edf,
        )
        phi_hat = float(np.sum(weights * (y - mu) ** 2 / var_mu)) / max(denom, 1.0)
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

        Built from the margin's EFFECTIVE spec on its resolved values, so an
        OrderedCategorical margin contributes its inner spline over the mapped
        level scores — column for column the marginal its refit would build.

        Evaluated directly on the compact support via the same
        ``support=``/``counts=`` path the discrete builder uses (centering
        direction ``counts @ basis`` equals the full-row column sums), so a
        million-row feature with a few dozen distinct values never
        materializes an (n, K) basis.
        """
        key = (name, binned)
        if key not in marginal_cache:
            s = _support(name, binned)
            m = TensorInteraction._marginal_from_spec(
                _margin_spec(name), s["x"], None, support=s["uniq"], counts=s["counts"]
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

        The budget gates run on support sizes alone, so an over-budget pair
        must never pay for a menu it is about to bin or skip away — which for
        a wide factor means never allocating its dense (L, L-1) contrast block.
        """
        if margin_kinds[name] == "categorical":
            return _levels_of(name)
        s = _support(name, binned)
        return s["codes"], s["n"]

    def _margin(name, binned):
        """(codes, support size, menu, penalty) for one margin.

        ``binned`` applies to spline margins only — a categorical margin's
        support IS its fitted level set, which quantile binning cannot and
        must not compress.  Its penalty is None: the contrast block is
        unpenalized, so there is nothing to smooth along that direction.
        """
        codes, n = _margin_support(name, binned)
        if margin_kinds[name] == "categorical":
            return codes, n, _contrast_of(name), None
        menu, S = _one_marginal(name, binned)
        return codes, n, menu, S

    def _pair_penalty(S_a, S_b, k_a, k_b):
        """The pair's block penalty, or None when no margin carries one.

        ``tensor_penalty(S, 0)`` is ``kron(S, I)``, which is exactly the
        block-diagonal per-level penalty a varying-coefficient refit applies
        (up to the column permutation the statistic is invariant to).
        """
        if S_a is None and S_b is None:
            return None
        return tensor_penalty(
            np.zeros((k_a, k_a)) if S_a is None else S_a,
            np.zeros((k_b, k_b)) if S_b is None else S_b,
        )

    def _within_budget(n_a, n_b, k_a, k_b):
        cells_ok = n_a * n_b <= max_cells
        inter_ok = n_a * k_b * k_b + n_b * k_a * k_a <= _INTERMEDIATE_BUDGET_FACTOR * max_cells
        return cells_ok and inter_ok

    def _numeric_margin_within_budget(k_g):
        """Budget a numeric x gridded pair BEFORE its menu is built.

        A z-moment pair allocates no cell grid, but it does allocate four
        blocks that all scale with the GRIDDED margin's width: the (L, L-1)
        menu, the (L-1)^2 curvature, the (L+1)^2 overlap curvature, and the
        cross block between them.  Holding the largest of those — the
        ``(k_g + 2)^2`` overlap block — to ``max_cells`` leaves the pair's
        total at roughly four ceiling-sized blocks, the same small multiple
        the two cell tables of a gridded pair sit at, and refuses the factors
        whose blocks would dwarf it (measured: ~160 MB at the threshold,
        ~640 MB at twice it).  Runs on the level count alone, so an
        over-budget pair never allocates the dense menu it was refused for.
        """
        return (k_g + 2) ** 2 <= max_cells

    def _within_cubic_budget(k, penalized):
        """Budget the pair's SOLVE TIME, which the allocation gates do not.

        Scoring a pair decomposes a ``(k, k)`` system, so per-pair time grows
        as ``k^3`` while the gates above grow as ``k^2`` — a block that fits
        memory comfortably can still take minutes.  A penalized block is
        charged ``_PENALIZED_LADDER_COST`` times the work: its ladder shares
        ONE decomposition across every rung, so it now costs barely more than
        an unpenalized block rather than the 25x a per-rung bisection did.
        Runs on the block dimension alone, before any menu or cell table is
        built, so a refused pair pays for nothing; the constants and the
        ~1.5 s per-pair target they were fitted to are documented at the top
        of this module.
        """
        work = (_PENALIZED_LADDER_COST if penalized else 1) * int(k) ** 3
        return work <= _CUBIC_BUDGET_FACTOR * max_cells

    def _within_structured_budget(k_s, n_levels):
        """Budget the BLOCK STACKS of a spline_cat pair scored through the
        arrow kernel.

        Nothing here is cubic in the level count and nothing is quadratic in
        it either, so this gate is linear where every gate above is not.  The
        pair allocates a handful of ``(n_levels, k_s + 1, k_s + 1)`` stacks.
        During issue #204's stable profiled-trace assembly, the persistent
        curvature stack coexists with one additive suffix stack; both are
        charged here.  The trace's centered-row temporaries are chunked and
        remain under the support intermediate gate below.  The suffix is
        discarded before the ladder allocates its blocks, inverses, border
        coupling and inverse diagonal blocks, so the two phases do not add
        their level stacks together.

        Charging two padded stack units against a factor of two retains the
        existing ``max_cells / (k_s + 1)^2`` level ceiling while making the
        extra live stack explicit instead of silently invalidating the
        advertised footprint.

        This bounds neither of the pair's other two costs.  The moment
        assembly's ``(n_a, k_s, k_s)`` intermediate is bounded by
        ``_within_structured_cells``, because it scales with the SUPPORT and
        binning can shrink it; the batched eigendecomposition is bounded by
        ``_structured_evaluation_budget``, because it is cubic in ``k_s``
        where this gate is quadratic.
        """
        live_stack_cells = 2 * int(n_levels) * (int(k_s) + 1) ** 2
        return live_stack_cells <= _STRUCTURED_BUDGET_FACTOR * max_cells

    def _within_structured_cells(n_a, n_levels, k_s):
        """Budget a structured pair's cell table and its curvature intermediate.

        The structured counterpart of ``_within_budget``, and the same two
        terms.  The kernel forms one ``(n_a, k_s, k_s)`` stack of the spline
        menu's outer products and contracts it against every level's weights
        — the TRANSPOSE of the intermediate the dense path is gated on, which
        no gate above bounds: ``_within_structured_budget`` holds
        ``n_levels * k_s^2`` and the cell term holds ``n_a * n_levels``, and
        neither is ``n_a * k_s^2``.  Measured on a width-45 spline over
        147,000 support points against a 34-level factor — a pair that passes
        both of those, the block-stack gate at 1/69th of its budget —
        ``screen_interactions`` peaked at 2,545 MB on that one pair, of which
        2,271 MB was this one array.

        Both terms scale with the SUPPORT, so failing either bins the spline
        margin first and refuses only when binning runs out, exactly as the
        dense path does; that same pair binned to 256 points peaks at 27 MB
        and is still scored, flagged ``approx``.
        """
        cells_ok = int(n_a) * int(n_levels) <= max_cells
        inter_ok = int(n_a) * int(k_s) ** 2 <= _INTERMEDIATE_BUDGET_FACTOR * max_cells
        return cells_ok and inter_ok

    def _structured_evaluation_budget(n_a, k_s, n_levels):
        """How many arrow factorizations a structured pair may spend.

        Budgets the pair's SOLVE TIME, which the allocation gates do not —
        the same job ``_within_cubic_budget`` does for a dense block, and for
        the same reason: one evaluation batches ``n_levels`` eigendecomposit-
        ions of ``(k_s + 1)`` blocks, so it costs ``n_levels * k_s^3`` where
        the gates above cost ``n_levels * k_s^2``.  A pair that cannot afford
        two of them cannot even bracket the ladder and is refused outright.

        Issue #204 adds work before those factorizations: two stable
        centered-row QR passes over the compressed cells, plus seven
        conservative ``(k_s + 1)^3`` units per level for the suffix/prefix
        QR merges, aligned representative QR, products and solve.  They cost
        ``2*n_a*n_levels*k_s^2 + 7*n_levels*(k_s+1)^3`` work units here.
        That setup is subtracted first, so a pair must still afford the two
        real endpoint factorizations after paying for its profiled trace; the
        count passed to ``structured_ladder`` continues to mean actual
        ``_evaluate`` calls.

        How many MORE evaluations it needs is not a function of its
        dimensions.  A rung
        whose budget lands inside the bracket bisects, at one factorization
        per step, and whether one does turns on the penalty's null space
        rather than on any size: measured on a 400-level pair, a ``ps``
        margin clamps every rung and the whole ladder is 2 evaluations, while
        an ``ns`` margin — whose penalty is full rank, so ``edf`` at maximum
        penalty is 0 and no rung can clamp — took 106.  Same dimensions, 53x
        the work.  So the ceiling is passed to the kernel, which brackets
        first, then checks the worst case for the rungs that genuinely have
        to search before spending anything on them.

        The setup charge also depends on ``n_a``, so an exact support that
        cannot afford it is allowed to reach spline binning and is retried on
        the compressed support.  A pair is refused only when the compressed
        setup plus two endpoints still exceeds the same work ceiling.
        """
        return _structured_evaluation_allowance(max_cells, n_a, k_s, n_levels)

    rows = []
    from superglm.dm_builder import resolve_discrete_n_bins, should_discretize

    # Discretization is decided per SPEC, not per model: spec.discrete
    # overrides the model flag.  Discretization is LOSSLESS when a margin's
    # cardinality fits its resolved bin count (the binner returns the exact
    # unique support), so approx flags only a refit whose basis genuinely
    # differs from the probe: some margin the refit would bin, binned lossily.
    model_discrete = bool(getattr(model, "_discrete", False))
    n_bins_config = getattr(model, "_n_bins", 256)

    def _refit_binned_margins(kind, feat_a, feat_b):
        """The margins whose basis the confirmatory refit would discretize.

        Mirrors the gates the builder itself applies, which differ by class: a
        ti() refit bins its marginal supports only when BOTH parents
        discretize, while a SplineCategorical refit bins its spline margin
        whenever that ONE parent does.  A categorical margin has nothing to
        compress and never appears here: only a margin that grids on a spline
        can put a refit out of step with the probe, and the OC-parented pairs
        that could are already short-circuited by the caller.
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
        # An OrderedCategorical parent refuses fit-time discretization outright,
        # whatever the model flag says and whatever its inner spline would say
        # alone (should_discretize_tensor_interaction and its spline_cat sibling
        # both gate on the parent spec): an OC margin already lives on at most
        # n_levels score points, so the refit has nothing to compress and always
        # sees the probe's own basis.
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
        # it None and is scored from the dense moments as before.
        structured_results = None
        if kind in ("numeric_cat", "numeric_numeric"):
            # A numeric margin enters the probe LINEARLY, so the pair has no
            # joint grid to assemble: z-weighted moments over the other
            # margin's cells are the exact sufficient statistics, with nothing
            # to bin and no cell grid to budget.  The block carries no penalty
            # either, so one rung is the ladder.  What is left to budget is the
            # OTHER margin's own blocks, which is why a factor too wide for
            # them is refused below — refusal is the only degradation a
            # z-moment pair has, since it cannot approximate.
            S_ti, approx = None, False
            if kind == "numeric_numeric":
                U, V, C, M, u_m = numeric_numeric_moments(
                    _raw_numeric(feat_a), _raw_numeric(feat_b), score, working_weights
                )
                # Two numerics contract to 3x3 blocks whatever the supports.
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
                U, V, C, M, u_m = numeric_pair_moments(
                    codes_g, n_g, menu_g, _raw_numeric(num_name), score, working_weights
                )
        else:
            # Assemble with the categorical margin LAST, so a mixed pair's penalty
            # is kron(S_spline, I) — the varying-coefficient block layout.  The
            # reported feature_a/feature_b keep the caller's order; the statistic
            # is invariant to the column permutation the swap amounts to.
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
            # Set when that handoff was taken speculatively, on a width
            # estimate that is biased low: if the true width then puts the
            # pair outside the arrow budgets, the dense path gets its binning
            # fallback back rather than the pair becoming a NaN row.
            arrow_lookahead = False
            # Latched once the LADDER itself refuses, as opposed to a gate.
            # Without it the restore below and the speculation further down
            # chase each other: the dense path fails its budget, speculates,
            # the ladder refuses, the dense track is handed back, and the same
            # speculation is taken again.
            arrow_refused = False
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
                # But a spline x categorical pair never has to form that block.
                # Grouped by level its bordered system is an arrow matrix — the
                # levels touch each other nowhere, only the intercept and the
                # spline main — which factorizes in time and memory LINEAR in
                # the level count.  So a pair the dense gates refuse gets
                # retried against the structured gate instead of skipped.
                structured = kind == "spline_cat" and not dense_ok
                structured_setup_ok = True
                if not (dense_ok or structured):
                    break
                if structured:
                    # The block stacks do not depend on support, so binning
                    # cannot rescue them.  The profiled-trace setup now DOES:
                    # its two centered-geometry passes scale with n_l, so an
                    # unaffordable exact support must reach the ordinary
                    # spline-binning fallback below rather than terminate.
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
                        # The ladder is run HERE, inside the routing loop,
                        # because whether the arrow path can score this pair is
                        # a routing question and not merely a scoring one.  Its
                        # cost turns on the penalty's null space rather than on
                        # any dimension, so no gate above can predict it and it
                        # can only refuse once tried -- and a speculative
                        # handoff that is refused must hand the dense track
                        # back, exactly as the width and support exits already
                        # do, rather than deleting a pair the dense path could
                        # still score.
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
                        )
                        if structured_results is None and arrow_lookahead:
                            allow_dense, arrow_lookahead = True, False
                            arrow_refused = True
                            continue
                        margins = ((menu_l, S_l), (level_rows, None))
                        break
                    _, _, menu_r, S_r = _margin(right, bin_flag[right])
                    if (menu_l.shape[1], menu_r.shape[1]) != (k_l, k_r):
                        # authoritative dims from the built menus; re-run the gates
                        k_l, k_r = menu_l.shape[1], menu_r.shape[1]
                        continue
                    margins = ((menu_l, S_l), (menu_r, S_r))
                    break
                # bin the largest not-yet-binned margin that binning can shrink;
                # a categorical margin is never binnable, whatever its level count
                binnable = sorted(
                    (
                        (n, name)
                        for n, name in ((n_l, left), (n_r, right))
                        if margin_kinds[name] == "spline" and not bin_flag[name] and n > screen_bins
                    ),
                    reverse=True,
                )
                if not binnable:
                    if arrow_lookahead:
                        # The speculative handoff has now run out of support
                        # fallback too, so the arrow path cannot score this
                        # pair either.  Hand the dense track back rather than
                        # dropping the pair: the handoff was taken to try for
                        # an EXACT score in place of an approximate one, and
                        # giving up the approximate one as well trades a
                        # scored row for a NaN.  Measured on a 10-level factor
                        # against a 6,000-point ps(8) support at
                        # max_cells=100_000, screen_bins=4_000: NaN here,
                        # against z=1.1374404130844136 once the dense track is
                        # restored.  The same restore the width recheck above
                        # already performs, at the other exit.
                        allow_dense, arrow_lookahead = True, False
                        continue
                    # The dense path has no moves left.  Hand a spline_cat
                    # pair to the arrow kernel before giving up: the width
                    # estimate that kept it on the dense track is biased LOW
                    # by design, so a pair whose true dimension would have
                    # routed it structurally can reach here still believing
                    # it was dense-affordable.
                    if (
                        allow_dense
                        and not structured
                        and kind == "spline_cat"
                        and not arrow_refused
                    ):
                        allow_dense = False
                        continue
                    break
                if allow_dense and not structured and kind == "spline_cat":
                    # About to compress the spline margin, which means the
                    # dense path can no longer score this pair EXACTLY.  The
                    # arrow path may still be able to, on the support as it
                    # stands — its intermediate is the transpose of the one
                    # that just failed, so failing that one says nothing about
                    # this one.  An exact score beats an approximate one, so
                    # try it before binning rather than after.  `left` is the
                    # spline margin for spline_cat, by the swap above.
                    if (
                        not arrow_refused
                        and _within_structured_budget(k_l, n_r)
                        and _structured_evaluation_budget(n_l, k_l, n_r) >= 2
                        and _within_structured_cells(n_l, n_r, k_l)
                    ):
                        allow_dense, arrow_lookahead = False, True
                        continue
                bin_flag[binnable[0][1]] = True
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
                    # The ladder can refuse for a search that exceeds its
                    # evaluation allowance, or for a numerical rank/EDF
                    # certificate that is not trustworthy enough to publish.
                    # Either is refused the way an unaffordable dense block
                    # is: a NaN row.  Reaching here means the dense track was
                    # already exhausted, since a SPECULATIVE handoff hands it
                    # back above instead.
                    rows.append(
                        (feat_a, feat_b, kind, np.nan, np.nan, np.nan, np.nan, n_cells, approx)
                    )
                    continue
            else:
                S_cell, W_cell = pair_cell_moments(
                    codes_l, codes_r, n_l, n_r, score, working_weights, max_cells=max_cells
                )
                U, V = pair_score_curvature(menu_l, menu_r, S_cell, W_cell)
                M, C, u_m = pair_overlap_moments(menu_l, menu_r, S_cell, W_cell)
                S_ti = _pair_penalty(S_l, S_r, menu_l.shape[1], menu_r.shape[1])
        # An unpenalized block has no bandwidth to scan: every rung returns the
        # same achieved rank, statistic and lambda0=0, so one rung is the ladder.
        penalized = structured_results is None and S_ti is not None and bool(np.any(S_ti))
        # The whole ladder shares ONE decomposition.  The pencil that turns
        # edf(lambda) and T(lambda) into closed forms depends on neither
        # lambda nor edf0, so every rung after the first costs O(k) instead of
        # a fresh O(k^3) solve — and the bisection that used to re-solve ~27
        # times per rung now runs on the closed form.
        #
        # This also retires the clamped-rung skip that used to guard the same
        # cost.  Its own justification was that a rung clamping UPWARD returns
        # an identical (statistic, edf0, lambda0) triple for every strictly
        # lower budget; an identical triple gives an identical z, and the
        # comparison below is STRICT, so recomputing those rungs cannot
        # displace the incumbent.  Same output, one less special case.
        results = structured_results
        if results is None:
            results = penalized_score_statistic_ladder(
                U,
                V,
                C,
                M,
                S_ti,
                budgets=tuple(float(b) for b in (budgets if penalized else budgets[:1])),
                U_nuisance=u_m,
            )
        best_z, best = -np.inf, None
        for result in results:
            if not result.edf0 > 0.0:
                # A rung that resolved NO direction at all has no test to run,
                # and the normalization divides by sqrt(2 * edf0) -- so scoring
                # it would report z = inf and sort a pair carrying no
                # information to the TOP of the table.  Skipped, and if no rung
                # survives the pair falls through to the NaN row every other
                # refusal takes.
                #
                # A pure arithmetic guard, deliberately: rank 0 means the
                # kernel resolved nothing, not that the interaction looked
                # weak.  Nothing upstream discards a pair for being weakly
                # identified -- see the THRESHOLD TYPES note in
                # superglm.screening._score_stat -- so this fires only on a
                # block with no resolvable direction whatsoever.
                continue
            statistic = result.statistic / phi_hat
            z = (statistic - result.edf0) / np.sqrt(2.0 * result.edf0)
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
