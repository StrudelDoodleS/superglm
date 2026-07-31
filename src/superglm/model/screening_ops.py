"""PSST interaction screening over a fitted mains model.

PSST — Penalized Smooth Score Test — ranks every candidate pair by how much
of the fitted model's leftover working signal the interaction block the pair
would actually refit as could absorb at a fixed screening complexity.  One
fused O(n) cell pass per pair, no refits; the confirmatory ``fit_reml``
refit of the top-ranked pairs is the gate.  Ranking-only: the statistic is
not a calibrated p-value and must not be reported as one.

The sweep is not spline-only.  The ``kind`` column names the interaction
class each pair would refit as — ``ti`` (spline x spline), ``spline_cat``,
and ``cat_cat`` today, with ``numeric_cat`` and ``numeric_numeric`` arriving
later in this release series.  Every kind runs through the same cell kernels
on a per-margin ``(codes, support size, menu, penalty)`` description: a
categorical margin is a gridded margin over its fitted levels whose menu is
the (L, L-1) treatment-contrast block and whose penalty is absent, so
``kron`` of two such menus reproduces the cross-product indicator columns and
``kron`` of a spline menu with one reproduces the per-level curve blocks,
column for column.  ``z`` normalizes each kind against its own noise floor,
so a single sorted table ranks them together.  A pair with no penalty
anywhere in its block has no bandwidth to scan and is evaluated at a single
rung: ``edf0`` then reports the block's achieved rank and ``lambda0`` is 0.

The statistic is reported on the ``T / phi`` scale, with ``phi`` the mains
fit's Pearson dispersion estimate: under the null ``E[T] = phi * edf0``, so
without this scaling the ``edf0`` noise floor is only honest for
unit-dispersion families and a dispersed Gaussian null would swamp the scan.
The Pearson denominator is ``n - edf`` per the sample_weight=exposure
contract (Var(y) = phi*V(mu)/w), deliberately NOT the frequency-weight
``sum(w) - edf`` convention of ``solvers.dispersion`` — measured on
known-dispersion exposure nulls, only ``n - edf`` is calibrated.

Per-feature work (unique support, codes, marginal basis menus, marginal
penalties — level codes and the contrast menu for a categorical margin) is
cached across the sweep, so an all-pairs screen over P screened features
builds each feature's marginal exactly once.  The caches trade
memory for time and live for the whole sweep: roughly the raw covariates
plus codes plus menus per screened feature (plus binned variants where the
fallback fires), instead of per-pair transients.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from superglm._frame import as_eager_frame
from superglm.distributions import _VARIANCE_FLOOR, validate_response
from superglm.features.categorical import (
    Categorical,
    _grouping_labels,
    _validate_categorical_levels,
)
from superglm.features.numeric import Numeric
from superglm.features.ordered_categorical import OrderedCategorical
from superglm.features.spline import _SplineBase
from superglm.screening import (
    pair_cell_moments,
    pair_score_curvature,
    penalized_score_statistic,
    working_score,
)
from superglm.screening._overlap import pair_overlap_moments, tensor_penalty

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
        # A spline-mode OC is a spline through the level values, so it screens
        # (and refits) exactly like one; step mode has no interaction target.
        if spec.basis == "spline" and spec._spline is not None:
            return "spline"
        return None
    if isinstance(spec, Categorical):
        return "categorical" if len(spec._levels) >= 2 else None
    if isinstance(spec, Numeric):
        return "numeric"
    return None


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


def _validated_pairs(candidates, margin_kinds, fitted_pairs, fitted_names):
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
            # OrderedCategorical, a one-level Categorical) is deferred, not a
            # typo; listing the screenable features would send the caller
            # hunting for a misspelling that isn't there.
            deferred_names = sorted(
                name for name in pair if name not in margin_kinds and name in fitted_names
            )
            if deferred_names:
                raise ValueError(
                    f"candidates entry {raw!r} names fitted feature(s) "
                    f"{deferred_names} that have no screenable margin — "
                    f"{_DEFERRED_KIND_HINT}"
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
    x = np.asarray(x_raw).ravel()
    if spec._grouping is not None:
        x = _grouping_labels(x)
        _validate_categorical_levels(x, set(spec._grouping.all_original_levels))
        x = pd.Series(x).map(spec._grouping.original_to_group).to_numpy()
    else:
        _validate_categorical_levels(x, set(spec._levels))
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


def _marginal_width_estimate(spec) -> int:
    """Guard-grade marginal width from the parent spline's geometry.

    Deliberately biased LOW: an under-estimate self-heals (menus get built
    and the authoritative post-menu recheck re-runs the gates with true
    dimensions), while an over-estimate would bin or skip a pair that fits
    the budget with no correction possible.  ``n_knots`` floors the centered
    marginal width of every built-in kind, including degree-0 ps/bs
    (``n_knots + degree``) and cr via the CardinalCR substitution
    (``n_knots + 1``).  Categorical and numeric widths are exact, not
    estimates: a contrast menu is (L - 1) wide and a numeric margin is 1.
    """
    if isinstance(spec, OrderedCategorical) and spec._spline is not None:
        spec = spec._spline
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
    spline crossed with a factor, ``cat_cat`` for two factors, with
    ``numeric_cat`` and ``numeric_numeric`` arriving later in this release
    series.  ``z`` normalizes each kind against its own noise floor, so one
    sorted table ranks them together.  A kind whose block carries no penalty
    (``cat_cat``) has no bandwidth to scan and is evaluated at a single
    rung — ``edf0`` then reports the block's achieved rank and ``lambda0`` is
    0, so the ``edf0`` argument does not apply to it.

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
    overridden with ``phi=`` (e.g. a frequency-weight user supplying
    ``sum(w) - edf`` semantics).

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
    always ``approx=False``.
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
    # Every interaction class exposes parent_names, so exclusion covers the
    # whole family (tensor, spline_cat, numeric_cat, cat_cat, ..., and
    # FactorSmooth) rather than tensor terms alone.
    fitted_pairs = {
        frozenset(spec.parent_names)
        for spec in getattr(model, "_interaction_specs", {}).values()
        if hasattr(spec, "parent_names")
    }
    pairs = _validated_pairs(candidates, margin_kinds, fitted_pairs, set(model._specs))

    def _is_oc_margin(name):
        return isinstance(model._specs.get(name), OrderedCategorical)

    def _select_flag(name):
        # Only spline margins carry a double-penalty flag; an OC margin's
        # lives on its inner spline.
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

    for name in sorted({name for pair in pairs for name in pair}):
        # A categorical margin is read through its level labels, and an
        # OrderedCategorical margin through its mapped level values (a later
        # task); prefetching the latter as a float column here would trip the
        # finiteness cast on label data and mask the per-pair diagnosis below.
        if margin_kinds[name] == "categorical":
            _raw_object(name)
        elif not _is_oc_margin(name):
            _raw_numeric(name)

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

    # Residual d.f. is n - edf, NOT the frequency-weight form sum(w) - edf of
    # pearson_residual_degrees_of_freedom: the screen follows the library's
    # protected sample_weight=exposure contract, under which Var(y) =
    # phi*V(mu)/w and E[Pearson] = n*phi.  Measured on known-phi exposure
    # nulls, n - edf tracks truth exactly while sum(w) - edf misreads phi 1.0
    # as ~1.55 and demotes refit-confirmed structure (see the plan log,
    # dispersion-denominator entry).
    if phi is not None:
        phi_hat = float(phi)
    else:
        edf_mains = float(getattr(model._result, "effective_df", float("nan")))
        # Zero-weight rows contribute exactly zero to the Pearson numerator,
        # so only positive-weight observations count toward the residual d.f.
        # (the docstring's "n - edf" means exactly this n).
        n_eff = float(np.count_nonzero(weights))
        denom = n_eff - edf_mains if np.isfinite(edf_mains) else n_eff
        phi_hat = float(np.sum(weights * (y - mu) ** 2 / var_mu)) / max(denom, 1.0)
        phi_hat = max(phi_hat, float(np.finfo(np.float64).tiny))

    support_cache: dict[tuple[str, bool], dict] = {}
    marginal_cache: dict[tuple[str, bool], tuple[np.ndarray, np.ndarray]] = {}
    level_cache: dict[str, tuple[np.ndarray, int]] = {}
    contrast_cache: dict[str, np.ndarray] = {}

    def _support(name, binned):
        key = (name, binned)
        if key not in support_cache:
            x = _raw_numeric(name)
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
                model._specs[name], s["x"], None, support=s["uniq"], counts=s["counts"]
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
        compress and never appears here, which is also what keeps ``_support``
        (and its float cast) away from a label column.
        """
        if kind not in ("ti", "spline_cat"):
            return []
        splines = [
            name
            for name in (feat_a, feat_b)
            if margin_kinds[name] == "spline"
            and should_discretize(model._specs[name], model_discrete)
        ]
        if kind == "ti" and len(splines) < 2:
            return []
        return splines

    def _pair_refits_discrete(kind, feat_a, feat_b):
        return any(
            _support(name, False)["n"]
            > resolve_discrete_n_bins(name, model._specs[name], n_bins_config)
            for name in _refit_binned_margins(kind, feat_a, feat_b)
        )

    for feat_a, feat_b in pairs:
        # Dispatch before any raw-column or support access, so a deferred kind
        # surfaces as its own error rather than a dtype failure downstream.
        kind = _pair_kind(margin_kinds[feat_a], margin_kinds[feat_b])
        if kind not in ("ti", "spline_cat", "cat_cat"):
            raise NotImplementedError(f"screening kind {kind!r} lands in a later task")
        # An OC margin screens as a spline, but on its MAPPED level values;
        # reading them is a later task, so say so rather than letting the
        # label column fail a float cast three frames down.
        oc_margins = [name for name in (feat_a, feat_b) if _is_oc_margin(name)]
        if oc_margins:
            raise NotImplementedError(
                f"screening OrderedCategorical margins {oc_margins} lands in a later task"
            )
        # Assemble with the categorical margin LAST, so a mixed pair's penalty
        # is kron(S_spline, I) — the varying-coefficient block layout.  The
        # reported feature_a/feature_b keep the caller's order; the statistic
        # is invariant to the column permutation the swap amounts to.
        left, right = feat_a, feat_b
        if margin_kinds[left] == "categorical" and margin_kinds[right] != "categorical":
            left, right = right, left
        k_l = _marginal_width_estimate(model._specs[left])
        k_r = _marginal_width_estimate(model._specs[right])
        bin_flag = {left: False, right: False}
        margins = None
        while True:
            codes_l, n_l = _margin_support(left, bin_flag[left])
            codes_r, n_r = _margin_support(right, bin_flag[right])
            # V is (k_l*k_r)^2 doubles regardless of support: binning can't help
            if (k_l * k_r) ** 2 > _INTERMEDIATE_BUDGET_FACTOR * max_cells:
                break
            if _within_budget(n_l, n_r, k_l, k_r):
                _, _, menu_l, S_l = _margin(left, bin_flag[left])
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
                break
            bin_flag[binnable[0][1]] = True
        approx = bin_flag[left] or bin_flag[right] or _pair_refits_discrete(kind, feat_a, feat_b)
        n_cells = n_l * n_r
        if margins is None:
            rows.append((feat_a, feat_b, kind, np.nan, np.nan, np.nan, np.nan, n_cells, approx))
            continue

        (menu_l, S_l), (menu_r, S_r) = margins
        S_cell, W_cell = pair_cell_moments(
            codes_l, codes_r, n_l, n_r, score, working_weights, max_cells=max_cells
        )
        U, V = pair_score_curvature(menu_l, menu_r, S_cell, W_cell)
        M, C, u_m = pair_overlap_moments(menu_l, menu_r, S_cell, W_cell)
        S_ti = _pair_penalty(S_l, S_r, menu_l.shape[1], menu_r.shape[1])
        # An unpenalized block has no bandwidth to scan: every rung returns the
        # same achieved rank, statistic and lambda0=0, so one rung is the ladder.
        penalized = S_ti is not None and bool(np.any(S_ti))
        best_z, best = -np.inf, None
        for budget in budgets if penalized else budgets[:1]:
            result = penalized_score_statistic(U, V, C, M, S_ti, edf0=budget, U_nuisance=u_m)
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
    return table
