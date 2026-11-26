"""Coefficient table and basis detail builders for model summaries."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from functools import cache
from typing import Any, NamedTuple

import numpy as np
import scipy.spatial
from numpy.typing import NDArray

from superglm._weight_units import coefficient_covariance, standard_errors
from superglm.features.constraint import rules_of
from superglm.inference._metrics_design import MetricsDesign, factor_from_gram, weighted_moments
from superglm.inference._term_covariance import (
    Determinacy,
    contrast_determined,
    feature_se_from_cov,
    function_determined,
)
from superglm.inference._term_helpers import (
    _resolve_group_lambda,
    ordered_level_fit,
    spline_group_enrichment,
    spline_groups,
)
from superglm.inference.covariance import (
    BinaryScaledCovariance,
    augmented_group_block,
    augmented_group_diagonal,
    covariance_contrast_se,
    covariance_quadratic_form,
    covariance_selected_block,
    covariance_slope_view,
    covariance_wald_statistic,
)
from superglm.inference.summary import (
    _THIN_LEVEL_ROWS,
    _BasisDetailRow,
    _CoefRow,
    _compute_coef_stats,
    unenforced_rules_tag,
)
from superglm.solvers.rank import diagonal_of_square, selected_group_name_set
from superglm.types import GroupSlice


def term_test_bases(model) -> dict[str, NDArray]:
    """Constrained term bases for a scalar fit, retained with its compact covariance."""
    from superglm.stats.wood_pvalue import term_test_basis

    if model._dm is None:
        return model._fit_inference_info.get("term_test_bases", {})
    bases = {}
    for group, matrix in zip(model._groups, model._dm.group_matrices, strict=True):
        basis = term_test_basis(group, matrix)
        if basis is not None:
            bases[group.name] = basis
    return bases


def _thin_levels(
    codes: NDArray, n_levels: int, sample_weights: NDArray, weight_semantics: str | None
) -> NDArray:
    """Which levels of a categorical are thin: few rows, and little weight.

    A level is thin when it holds fewer than ``_THIN_LEVEL_ROWS`` rows AND less weight than that
    many rows of average weight; either many rows, or a large share of the book's weight however
    few rows carry it (an aggregated book of rating cells), is enough experience. Rows are
    counted as the weight contract counts them: under ``"frequency"`` a weight is a replication
    count (so the two conditions coincide); under ``"prior"`` (or no declared contract) a row is
    one observation and a zero weight an unobserved row, so a level's rows are its positive-weight
    rows and the average is over those. Neither condition depends on the book's size, the
    weights' units or the response. A level with ``_THIN_LEVEL_ROWS`` rows or more is never thin
    however little weight they carry.
    """
    from superglm.solvers.dispersion import FREQUENCY_WEIGHTS

    weights = np.asarray(sample_weights, dtype=np.float64)
    counted = weights if weight_semantics == FREQUENCY_WEIGHTS else (weights > 0.0) * 1.0
    rows = np.bincount(codes, weights=counted, minlength=n_levels + 1)[:n_levels]
    weight = np.bincount(codes, weights=weights, minlength=n_levels + 1)[:n_levels]
    rows_of_average_weight = weight * (np.sum(counted) / max(float(np.sum(weights)), 1e-300))
    return (rows < _THIN_LEVEL_ROWS) & (rows_of_average_weight < _THIN_LEVEL_ROWS)


def _ordered_piecewise_structural_rows(
    spec: Any,
    feature_name: Any,
    feature_label: str,
    b_main: NDArray,
    V_b_main: NDArray,
    alpha: float,
    *,
    is_sandwich: bool = False,
    sandwich_penalized: bool = False,
    wald_df_limit: int | None = None,
    determines: Callable[[NDArray], NDArray[np.bool_]] | None = None,
) -> list[_CoefRow]:
    """Structural contrast rows for an OrderedCategorical Piecewise basis.

    One slope-change Wald row per stated break and one curvature row per
    segment of degree >= 2 (a single z row for one curvature freedom, a joint
    chi-square family row for more): ordinary linear-model tests of a
    fixed-knot design (Smith 1979, Am. Statist. 33(2):57-62). Under C0 seams
    the segments share their joint values, so per-segment per-power z-tests
    do not exist. ``determines`` (over the block's coefficients: a robust
    covariance's, or the fit's ``RankDecomposition.is_estimable``) leaves a row
    it does not determine its estimate alone.
    """
    from scipy.stats import chi2 as chi2_dist

    inner = spec._basis_spline
    knots = inner._knots
    levels = spec._smooth_levels

    def _level_at(knot_index: int) -> str:
        return str(levels[int(round(float(knots[knot_index])))])

    def undetermined(contrasts: NDArray) -> bool:
        return determines is not None and not bool(np.all(determines(np.atleast_2d(contrasts))))

    rows: list[_CoefRow] = []
    for structural in inner.ordered_structural_rows():
        if structural.kind == "slope_change":
            contrast = structural.contrast
            estimate = float(contrast @ b_main)
            name = f"{feature_name}[slope-change @ {_level_at(structural.index)}]"
            if undetermined(contrast):
                rows.append(
                    _CoefRow(name=name, group=feature_label, coef=estimate, estimable=False)
                )
                continue
            se_val = covariance_contrast_se(V_b_main, contrast)
            z, p, ci_lo, ci_hi = _compute_coef_stats(estimate, se_val, alpha)
            rows.append(
                _CoefRow(
                    name=name,
                    group=feature_label,
                    coef=estimate,
                    se=se_val,
                    z=z,
                    p=p,
                    ci_low=ci_lo,
                    ci_high=ci_hi,
                )
            )
            continue
        segment = structural.index
        segment_label = f"{_level_at(segment)}..{_level_at(segment + 1)}"
        indices = np.asarray(structural.column_indices, dtype=np.intp)
        if indices.size == 1:
            coef_val = float(b_main[indices[0]])
            contrast = np.zeros(len(b_main))
            contrast[indices[0]] = 1.0
            if undetermined(contrast):
                rows.append(
                    _CoefRow(
                        name=f"{feature_name}[curvature {segment_label}]",
                        group=feature_label,
                        coef=coef_val,
                        estimable=False,
                    )
                )
                continue
            se_val = covariance_contrast_se(V_b_main, contrast)
            z, p, ci_lo, ci_hi = _compute_coef_stats(coef_val, se_val, alpha)
            rows.append(
                _CoefRow(
                    name=f"{feature_name}[curvature {segment_label}]",
                    group=feature_label,
                    coef=coef_val,
                    se=se_val,
                    z=z,
                    p=p,
                    ci_low=ci_lo,
                    ci_high=ci_hi,
                )
            )
            continue
        b_family = b_main[indices]
        if isinstance(V_b_main, BinaryScaledCovariance):
            V_family = BinaryScaledCovariance(
                V_b_main.base[np.ix_(indices, indices)], V_b_main.exponent
            )
        else:
            V_family = V_b_main[np.ix_(indices, indices)]
        stat: float | None = float("nan")
        p_val: float | None = float("nan")
        ref_df: float | None = float(indices.size)
        family_undetermined = undetermined(np.eye(len(b_main))[indices])
        if is_sandwich:
            if sandwich_penalized or family_undetermined:
                stat = p_val = ref_df = None
            else:
                from superglm.inference._sandwich_core import wald_block_statistic

                stat, p_val = wald_block_statistic(
                    b_family, np.asarray(V_family, dtype=np.float64), df_limit=wald_df_limit
                )
                if stat is None:
                    ref_df = None
        elif family_undetermined:
            stat = p_val = ref_df = None
        else:
            try:
                stat = covariance_wald_statistic(b_family, V_family)
                p_val = 1.0 - chi2_dist.cdf(stat, float(indices.size))
            except np.linalg.LinAlgError:
                pass
        rows.append(
            _CoefRow(
                name=f"{feature_name}[curvature {segment_label}]",
                group=feature_label,
                is_spline=True,
                n_params=int(indices.size),
                active=True,
                group_norm=float(np.linalg.norm(b_family)),
                wald_chi2=stat,
                wald_p=p_val,
                ref_df=ref_df,
                subgroup_type="curvature",
                estimable=not family_undetermined,
            )
        )
    return rows


def _selected_names(result: Any, groups: list[GroupSlice], selected_group_names) -> set[str]:
    """The groups the selected model keeps: the caller's set, else the fit's selection."""
    if selected_group_names is None:
        return selected_group_name_set(result, groups)
    return set(selected_group_names)


def _coefficient_mask(result: Any, override: NDArray | None) -> NDArray[np.bool_]:
    """Which coefficients show a standard error: the caller's mask, else the fit's rank record's."""
    if override is not None:
        return np.asarray(override, dtype=bool)
    if getattr(result, "rank_info", None) is not None:
        return result.rank_info.coefficient_estimable()
    return np.ones(len(result.beta), dtype=bool)


def _error_reader(result: Any, known_scale: bool, is_sandwich: bool) -> Callable[[Any], Any]:
    """Standard errors from variances: a robust covariance's own, else the fitted family's."""
    if is_sandwich:
        return lambda variance: np.sqrt(np.maximum(variance, 0.0))
    return lambda variance: standard_errors(variance, result, known_scale=known_scale)


def _level_diagnostics(
    groups: list[GroupSlice],
    group_matrices: list | None,
    sample_weights: NDArray | None,
    weight_semantics: str | None,
) -> dict[str, dict[int, tuple[int, float, bool]]]:
    """Each categorical level's row count, weight share and thinness (``_thin_levels``)."""
    from superglm.group_matrix import CategoricalGroupMatrix

    if group_matrices is None or sample_weights is None:
        return {}
    total_weight = float(np.sum(sample_weights))
    diagnostics = {}
    for gm, g in zip(group_matrices, groups):
        if not isinstance(gm, CategoricalGroupMatrix):
            continue
        K = gm.n_levels
        n_per = np.bincount(gm.codes, minlength=K + 1)[:K]
        exp_per = np.bincount(gm.codes, weights=sample_weights, minlength=K + 1)[:K]
        exp_share = exp_per / max(total_weight, 1e-300)
        thin = _thin_levels(gm.codes, K, sample_weights, weight_semantics)
        diagnostics[g.name] = {
            i: (int(n_per[i]), float(exp_share[i]), bool(thin[i])) for i in range(K)
        }
    return diagnostics


class _GroupTest(NamedTuple):
    """A whole-group test row's statistic, p-value and reference df."""

    stat: float | None
    p: float | None
    ref_df: float | None
    # The fit does not determine what the test is of, so the test is withheld.
    undetermined: bool = False


_WITHHELD = _GroupTest(None, None, None, undetermined=True)


@cache
def _row_producers() -> tuple[tuple[Any, Callable[..., list[_CoefRow]]], ...]:
    """Each feature class's row producer, in the order the classes are tested."""
    from superglm.features.categorical import Categorical
    from superglm.features.factor_smooth import FactorSmooth
    from superglm.features.interaction import (
        CategoricalInteraction,
        NumericCategorical,
        NumericInteraction,
        PolynomialCategorical,
        PolynomialInteraction,
        SplineCategorical,
        TensorInteraction,
    )
    from superglm.features.numeric import Numeric
    from superglm.features.ordered_categorical import OrderedCategorical
    from superglm.features.piecewise import Piecewise
    from superglm.features.polynomial import Polynomial
    from superglm.features.random_effect import RandomEffect
    from superglm.features.spline import _SplineBase

    return (
        (RandomEffect, _CoefTable.random_effect_rows),
        (FactorSmooth, _CoefTable.factor_smooth_rows),
        (OrderedCategorical, _CoefTable.ordered_rows),
        (_SplineBase, _CoefTable.spline_rows),
        (Categorical, _CoefTable.categorical_rows),
        (SplineCategorical, _CoefTable.spline_by_level_rows),
        (Polynomial, _CoefTable.polynomial_rows),
        (PolynomialCategorical, _CoefTable.polynomial_by_level_rows),
        (CategoricalInteraction, _CoefTable.crossed_level_rows),
        (NumericCategorical, _CoefTable.numeric_by_level_rows),
        (NumericInteraction | PolynomialInteraction, _CoefTable.numeric_interaction_rows),
        (TensorInteraction, _CoefTable.tensor_rows),
        (Numeric, _CoefTable.numeric_rows),
        (Piecewise, _CoefTable.piecewise_rows),
    )


@dataclass(kw_only=True, eq=False)
class _CoefTable:
    """One coefficient table: ``build_coef_rows``' arguments, field for field, and its rows.

    The per-call state (the fit, its covariance, the options) is read once here, and
    each feature class's rows come from one producer method (``_row_producers``).
    """

    groups: list[GroupSlice]
    specs: dict
    interaction_specs: dict
    result: Any
    X_a: Any
    W: NDArray
    XtWX_inv: NDArray
    XtWX_inv_aug: Any
    active_groups: list[GroupSlice]
    known_scale: bool
    group_edf_map: dict | None
    reml_lambdas: dict | None
    lambda2: float | dict
    n_obs: int
    alpha: float
    monotone_repairs: dict | None
    precomputed_R_a: NDArray | None  # noqa: N815 -- the argument of build_coef_rows
    precomputed_edf: NDArray | None
    precomputed_edf1: NDArray | None
    precomputed_design_moments: tuple[NDArray, NDArray, NDArray] | None
    test_bases: dict[str, NDArray] | None
    coefficient_estimable_override: NDArray | None
    selected_group_names: set[str] | None
    group_matrices: list | None
    sample_weights: NDArray | None
    weight_semantics: str | None
    recorded_likelihood_size: float | None
    selection_shrunk_group_names: set[str] | None
    rules_enforced: bool
    sandwich_covariance: Any | None
    contrast_estimable: Callable[..., bool] | None
    information_underflowed: bool | None
    intercept_centre: NDArray | None

    def __post_init__(self) -> None:
        from superglm.model.separated_levels import separated_coefficients

        result, sandwich = self.result, self.sandwich_covariance
        self.beta = result.beta
        self.selected = _selected_names(result, self.groups, self.selected_group_names)
        self.estimable = _coefficient_mask(result, self.coefficient_estimable_override)
        if self.contrast_estimable is None and self.coefficient_estimable_override is None:
            rank_info = getattr(result, "rank_info", None)
            self.contrast_estimable = None if rank_info is None else rank_info.is_estimable
        if self.information_underflowed is None:
            self.information_underflowed = bool(getattr(result, "information_underflowed", False))
        self.is_sandwich = sandwich is not None
        self.sandwich_penalized = (
            frozenset() if sandwich is None else frozenset(sandwich.penalized_group_names)
        )
        self.wald_df_limit = None if sandwich is None else sandwich.wald_df_limit
        # The slope blocks of the metrics a robust fit's level and curve errors are read
        # with (``UndeterminedDirections.metrics``), formed once, or none.
        self.undetermined = None if sandwich is None else sandwich.undetermined
        self.undetermined_metrics = (
            None
            if self.undetermined is None
            else tuple(metric[1:, 1:] for metric in self.undetermined.metrics())
        )
        self.read_errors = _error_reader(result, self.known_scale, self.is_sandwich)
        self.active_by_name = {group.name: group for group in self.active_groups}
        self.errors = {g.name: self._group_errors(g) for g in self.groups}
        # The coefficients a separated level or cell moves (``separated_coefficients``,
        # the intercept last): each keeps its estimate, with its Wald fields blank.
        self.separated = separated_coefficients(self.groups, self.specs, self.interaction_specs)
        self.level_diag = _level_diagnostics(
            self.groups, self.group_matrices, self.sample_weights, self.weight_semantics
        )
        self.repaired = self.monotone_repairs or {}
        self._moments = self.precomputed_design_moments
        self._R = self.precomputed_R_a
        self._influence = (
            None
            if self.precomputed_edf is None or self.precomputed_edf1 is None
            else (self.precomputed_edf, self.precomputed_edf1)
        )
        self._group_edf = self.group_edf_map
        self._ordered_done: set[Hashable] = set()

    def rows(self) -> list[_CoefRow]:
        """The intercept row, then each group's rows from its feature class's producer."""
        producers = _row_producers()
        rows = [self.intercept_row()]
        for g in self.groups:
            spec = self.specs.get(g.feature_name) or self.interaction_specs.get(g.feature_name)
            produce = next(
                (method for kinds, method in producers if isinstance(spec, kinds)),
                _CoefTable.default_rows,
            )
            rows.extend(produce(self, g, spec))
        # Not every producer sets ``estimable``; a blank error always means it is False.
        for row in rows:
            if row.se is not None and np.isnan(row.se):
                row.estimable = False
        _flag_outsized_errors(rows)
        return rows

    # ── Errors and determinacy ────────────────────────────────────

    def _group_errors(self, g: GroupSlice) -> NDArray:
        """A group's coefficient standard errors, NaN where the fit does not determine one."""
        if g.name not in self.selected:
            return np.zeros(g.size)
        active_group = self.active_by_name.get(g.name)
        errors = (
            np.zeros(g.size)
            if active_group is None
            else self.read_errors(augmented_group_diagonal(self.XtWX_inv_aug, active_group))
        )
        errors = errors.astype(float, copy=True)
        errors[~self.estimable[g.sl]] = np.nan
        return errors

    def cov_from_aug(self, matrix: Any) -> Any:
        """A block of the augmented inverse as a coefficient covariance."""
        if self.is_sandwich:
            return matrix
        return coefficient_covariance(matrix, self.result, known_scale=self.known_scale)

    def group_block(self, active_group: GroupSlice) -> NDArray:
        """An active group's coefficient covariance block."""
        return self.cov_from_aug(augmented_group_block(self.XtWX_inv_aug, active_group))

    def coef_stats(self, coef: float, se: float, group_name: str):
        """z, p and interval of one coefficient; a penalised block's robust z and p are blank."""
        z, p, ci_lo, ci_hi = _compute_coef_stats(coef, se, self.alpha)
        if self.is_sandwich and group_name in self.sandwich_penalized:
            return None, None, ci_lo, ci_hi
        return z, p, ci_lo, ci_hi

    def moves(self, column: int) -> bool:
        return self.separated is not None and bool(self.separated[column])

    def determined(self, contrast: NDArray, intercept: float = 0.0) -> bool:
        """Whether the default covariance determines ``contrast @ beta`` (``contrast_determined``)."""
        return contrast_determined(contrast, self.estimable, self.contrast_estimable, intercept)

    def mask_determines(self, g_slice: slice) -> Callable[[NDArray], NDArray[np.bool_]]:
        """Which contrasts of the coefficients at ``g_slice`` the default covariance determines."""

        def determined(contrast: NDArray) -> bool:
            full = np.zeros(len(self.beta))
            full[g_slice] = contrast
            return self.determined(full)

        return lambda contrasts: np.array([determined(c) for c in contrasts], dtype=bool)

    def function_determined(self, term_groups: list[GroupSlice]) -> bool:
        """Whether the default covariance determines the fitted function of ``term_groups``' blocks.

        Wood's test is of that function: each design row's part on the blocks' coefficients
        is one of its contrasts (``function_determined``).
        """
        columns = np.zeros(len(self.beta), dtype=bool)
        for group in term_groups:
            columns[group.sl] = True
        return function_determined(columns, self.estimable, self.contrast_estimable)

    def block_determines(self, g: GroupSlice) -> Callable[[NDArray], NDArray[np.bool_]]:
        """Which contrasts of active group ``g``'s coefficients the covariance in hand determines.

        The default covariance decides by its mask's rule (``mask_determines``).  A robust
        one has the same first clause, a contrast of coefficients its mask all shows, and
        decides the rest by its bread (``SandwichCovariance.determines``): a truncated
        structured bread keeps the mask and no null direction to decide by.
        """
        covariance = self.sandwich_covariance
        if covariance is None:
            return self.mask_determines(g.sl)
        start = 1 + self.active_by_name[g.name].start
        hidden = ~self.estimable[g.sl]

        def determines(contrasts: NDArray) -> NDArray[np.bool_]:
            rows = np.zeros((len(contrasts), covariance.n_params))
            rows[:, start : start + contrasts.shape[1]] = contrasts
            shown = ~np.any(contrasts[:, hidden] != 0.0, axis=1)
            return shown | covariance.determines(rows)

        return determines

    def feature_se(self, feature_name: Hashable) -> NDArray:
        """A term's level or curve errors from the slope covariance (``feature_se_from_cov``).

        Each one whose contrast the fit does not determine is blank.  The default
        covariance decides where the map is formed, by its mask's rule
        (``feature_se_from_cov``'s ``determines``); a robust one reads the term's own map
        with the covariance replaced by each metric of ``UndeterminedDirections``, which
        gives every row's share along the undetermined directions.
        """

        def read(covariance: Any, determines: Determinacy | None = None) -> NDArray:
            return np.asarray(
                feature_se_from_cov(
                    feature_name,
                    covariance,
                    self.active_groups,
                    self.result,
                    self.groups,
                    self.specs,
                    self.interaction_specs,
                    determines=determines,
                ),
                dtype=np.float64,
            )

        determines = (
            None if self.is_sandwich else Determinacy(self.estimable, self.contrast_estimable)
        )
        se = read(self.cov_from_aug(covariance_slope_view(self.XtWX_inv_aug)), determines)
        if self.undetermined is None or self.undetermined_metrics is None:
            return se
        norms = [read(metric) for metric in self.undetermined_metrics]
        return np.where(self.undetermined.determined(*norms), se, np.nan)

    def curve_se_range(self, feature_name: Hashable) -> tuple[float, float]:
        """The smallest and largest error on a spline term's curve.

        A specials term's errors include its FREE levels, whose uncertainty is not a
        point on the curve; levels are ordered smooth-then-special (the build()
        contract), so the tail is dropped.
        """
        se_curve = self.feature_se(feature_name)
        n_special = len(getattr(self.specs.get(feature_name), "_specials", ()) or ())
        if n_special and len(se_curve) > n_special:
            se_curve = np.asarray(se_curve)[:-n_special]
        return float(np.min(se_curve)), float(np.max(se_curve))

    # ── Whole-group tests ─────────────────────────────────────────

    def untested(self, size: int) -> _GroupTest:
        """A group test not computed: blank under a robust covariance, NaN on ``size`` df otherwise."""
        if self.is_sandwich:
            return _GroupTest(None, None, None)
        return _GroupTest(float("nan"), float("nan"), float(size))

    def block_wald(self, g: GroupSlice, V_block: NDArray) -> _GroupTest:
        """The chi-square of ``beta = 0`` on active group ``g``'s unpenalised block.

        Each coefficient is a tested contrast: the test is withheld unless the covariance
        in hand determines every one (``block_determines``).
        """
        from scipy.stats import chi2 as chi2_dist

        b_block, ref_df = self.beta[g.sl], float(g.size)
        if not np.all(self.block_determines(g)(np.eye(g.size))):
            return _WITHHELD
        if self.is_sandwich:
            return self._robust_block_wald(b_block, V_block, g.sl, g.name, ref_df)
        try:
            stat = covariance_wald_statistic(b_block, V_block)
            return _GroupTest(stat, 1.0 - chi2_dist.cdf(stat, ref_df), ref_df)
        except np.linalg.LinAlgError:
            return _GroupTest(float("nan"), float("nan"), ref_df)

    def _robust_block_wald(self, b_block, V_block, g_slice, group_name, ref_df) -> _GroupTest:
        from superglm.inference._sandwich_core import wald_block_statistic

        # The penalty that voids an ordinary Wald test is the one on this block.
        if (
            group_name in self.sandwich_penalized
            or not np.all(self.estimable[g_slice])
            or any(self.moves(idx) for idx in range(g_slice.start, g_slice.stop))
        ):
            return _GroupTest(None, None, None)
        stat, p = wald_block_statistic(b_block, V_block, df_limit=self.wald_df_limit)
        return _GroupTest(None, None, None) if stat is None else _GroupTest(stat, p, ref_df)

    def group_wald_test(self, g: GroupSlice) -> _GroupTest:
        """The chi-square on an active group's whole block (``block_wald``)."""
        return self.block_wald(g, self.group_block(self.active_by_name[g.name]))

    def wood_test(
        self,
        term_groups: list[GroupSlice],
        coef: NDArray,
        columns: slice | NDArray,
        covariance: NDArray,
        size: int,
        *,
        catch: type[Exception],
    ) -> _GroupTest:
        """Wood's (2013) test of ``term_groups``' fitted function, at ``columns`` of the active design.

        Withheld where the default covariance does not determine that function
        (``function_determined``); ``catch`` is the failure that leaves the row untested.
        """
        if not self.function_determined(term_groups):
            return _WITHHELD
        R_a = self.R_factor()
        edf, edf1 = self.influence_edf()
        rank = float(np.sum(edf1[columns]))
        design = R_a[:, columns]
        residual_df = self.wood_residual_df(edf)
        try:
            return _GroupTest(
                *self.wood_statistic(term_groups, coef, design, covariance, rank, residual_df)
            )
        except catch:
            return self.untested(size)

    def group_wood_test(self, g: GroupSlice) -> _GroupTest:
        """Wood's test of one active group's function; a robust covariance reports none."""
        if self.is_sandwich:
            return self.untested(g.size)
        active_group = self.active_by_name[g.name]
        return self.wood_test(
            [g],
            self.beta[g.sl],
            active_group.sl,
            self.group_block(active_group),
            g.size,
            catch=Exception,
        )

    def wood_statistic(self, term_groups, coef, design, covariance, rank, residual_df):
        from scipy.linalg import block_diag

        from superglm.stats.wood_pvalue import wood_test_smooth

        test_bases = self.test_bases
        if not test_bases or not any(g.name in test_bases for g in term_groups):
            return wood_test_smooth(coef, design, covariance, rank, residual_df)
        bases = [
            test_bases[g.name] if g.name in test_bases else np.eye(g.size) for g in term_groups
        ]
        basis = bases[0] if len(bases) == 1 else block_diag(*bases)
        return wood_test_smooth(coef, design, covariance, rank, residual_df, term_basis=basis)

    def wood_residual_df(self, edf: NDArray) -> float:
        if self.known_scale:
            return -1.0
        effective_df = 1.0 + float(np.sum(edf))
        if self.weight_semantics is not None and self.sample_weights is not None:
            from superglm.solvers.dispersion import pearson_residual_degrees_of_freedom

            return pearson_residual_degrees_of_freedom(
                self.sample_weights,
                effective_df,
                weight_semantics=self.weight_semantics,
            )
        # A released fit (``retain_fit_state=False``) has no weight rows left, but the fit
        # recorded its own likelihood size before discarding them: ``n_obs`` would be the
        # row count for a prior-contract fit carrying zero weights, and ``sum(w)`` is not
        # the row count under ``"frequency"`` either.
        if self.recorded_likelihood_size is not None:
            return max(float(self.recorded_likelihood_size) - effective_df, 1.0)
        return max(float(self.n_obs) - effective_df, 1.0)

    # ── Lazily formed design quantities (only Wood's test and EDF read them) ──

    def design_moments(self):
        if self._moments is None:
            self._moments = weighted_moments(self.X_a, self.W)
        return self._moments

    def R_factor(self) -> NDArray:
        if self._R is None:
            self._R = (
                np.empty((0, 0))
                if self.X_a.shape[1] == 0
                else factor_from_gram(self.design_moments()[2])
            )
        return self._R

    def influence_edf(self) -> tuple[NDArray, NDArray]:
        if self._influence is None:
            if self.X_a.shape[1] == 0:
                self._influence = (np.array([]), np.array([]))
            else:
                F = self.XtWX_inv_aug[1:, 1:] @ self.design_moments()[2]
                edf = np.diag(F)
                self._influence = (edf, 2.0 * edf - diagonal_of_square(F))
        return self._influence

    def group_edf(self) -> dict[str, float]:
        if self._group_edf is None:
            edf, _ = self.influence_edf()
            self._group_edf = {ag.name: float(np.sum(edf[ag.sl])) for ag in self.active_groups}
        return self._group_edf

    def mapped_edf(self, group_name: str) -> float | None:
        """A group's EDF from the group map, or ``None`` when the map is empty."""
        edf_map = self.group_edf()
        return edf_map.get(group_name) if edf_map else None

    # ── Row metadata ──────────────────────────────────────────────

    def spline_fields(self, group_name: str, spec: Any) -> dict[str, Any]:
        """A spline row's EDF, smoothing lambda and basis description (``spline_group_enrichment``)."""
        return spline_group_enrichment(
            group_name, spec, self.group_edf(), self.reml_lambdas, self.lambda2
        )

    def rule_fields(self, spec: Any, g: GroupSlice) -> dict[str, Any]:
        """A spline term's rule line: its rules, the engine, and whether a repair moved it."""
        return {
            "monotone": rules_of(spec).describe() or None,
            "monotone_engine": self.engine_tag(spec, g.monotone_engine),
            "monotone_repaired": g.feature_name in self.repaired,
        }

    def engine_tag(self, spec: Any, engine: str | None) -> str | None:
        """The engine on a term's rule line, or the mark that a path did not enforce its rules."""
        if engine is None or self.rules_enforced:
            return engine
        return unenforced_rules_tag(rules_of(spec))

    def structured_lambdas(self, group_name: str) -> tuple[tuple[str, float], ...]:
        source = self.reml_lambdas if self.reml_lambdas is not None else self.lambda2
        if not isinstance(source, dict):
            return (("lambda", float(source)),)
        exact = (("lambda", float(source[group_name])),) if group_name in source else ()
        prefix = f"{group_name}:"
        components = tuple(
            (name[len(prefix) :], float(value))
            for name, value in source.items()
            if name.startswith(prefix)
        )
        return exact + components

    # ── Row shapes ────────────────────────────────────────────────

    def coefficient_row(
        self,
        g: GroupSlice,
        i: int,
        name: str,
        group: str,
        *,
        blank: bool = False,
        withheld: bool = False,
        **fields: Any,
    ) -> _CoefRow:
        """Coefficient ``i`` of group ``g`` with its standard error, z, p and interval.

        ``blank`` (a separated level or cell) keeps only the estimate; ``withheld``
        keeps the error but not z, p or the interval.
        """
        coef = float(self.beta[g.start + i])
        if blank:
            return _CoefRow(name=name, group=group, coef=coef, separated=True, **fields)
        se = float(self.errors[g.name][i])
        z, p, ci_lo, ci_hi = (None,) * 4 if withheld else self.coef_stats(coef, se, g.name)
        return _CoefRow(
            name=name,
            group=group,
            coef=coef,
            se=se,
            z=z,
            p=p,
            ci_low=ci_lo,
            ci_high=ci_hi,
            **fields,
        )

    def group_row(self, g: GroupSlice, test: _GroupTest | None, **fields: Any) -> _CoefRow:
        """Group ``g``'s whole-term test row: active with ``test``, or inactive (``None``)."""
        if test is None:
            return _CoefRow(
                name=g.name,
                group=str(g.feature_name),
                is_spline=True,
                n_params=g.size,
                active=False,
                group_norm=0.0,
                **fields,
            )
        return _CoefRow(
            name=g.name,
            group=str(g.feature_name),
            is_spline=True,
            n_params=g.size,
            active=True,
            group_norm=float(np.linalg.norm(self.beta[g.sl])),
            wald_chi2=test.stat,
            wald_p=test.p,
            ref_df=test.ref_df,
            estimable=not test.undetermined,
            **fields,
        )

    # ── The intercept ─────────────────────────────────────────────

    def intercept_row(self) -> _CoefRow:
        """The intercept at each ordered term's base level, the effects being reported there.

        The same affine shift is applied to its covariance.  On the default path the
        reported intercept, the centred one less the centre and plus the ordered terms'
        base-level effects, is decided by the mask's rule as that one estimate: the centred
        intercept's unit weight is part of its size, so a centre of rounding size beside it
        decides nothing.
        """
        from superglm.inference._ordered_reference import (
            ordered_reference_beta_contrast,
            ordered_reference_intercept,
        )

        beta, groups, specs = self.beta, self.groups, self.specs
        feature_order = list(specs)
        intercept = ordered_reference_intercept(
            self.result.intercept, beta, feature_order, specs, groups
        )
        full_contrast = ordered_reference_beta_contrast(len(beta), feature_order, specs, groups)
        augmented_contrast = np.zeros(self.XtWX_inv_aug.shape[0], dtype=np.float64)
        augmented_contrast[0] = 1.0
        original_groups = {group.name: group for group in groups}
        for active_group in self.active_groups:
            original_group = original_groups[active_group.name]
            augmented_contrast[1 + active_group.start : 1 + active_group.end] = full_contrast[
                original_group.sl
            ]
        icpt_se = float(
            self.read_errors(covariance_quadratic_form(self.XtWX_inv_aug, augmented_contrast))
        )
        # Blank where the fit does not determine the base-level contrast, not only the
        # unadjusted intercept.
        sandwich = self.sandwich_covariance
        if sandwich is not None and not sandwich.determines(augmented_contrast)[0]:
            icpt_se = float("nan")
        if self.information_underflowed:
            icpt_se = float("nan")
        z, p, ci_lo, ci_hi = _compute_coef_stats(intercept, icpt_se, self.alpha)
        blank = self.moves(-1)
        selected_columns = np.zeros(len(beta), dtype=bool)
        for group in groups:
            selected_columns[group.sl] = group.name in self.selected
        contrast = full_contrast
        if self.intercept_centre is not None:
            contrast = contrast - np.asarray(self.intercept_centre, dtype=np.float64)
        undetermined = not self.is_sandwich and not self.determined(
            np.where(selected_columns, contrast, 0.0),
            intercept=0.0 if self.intercept_centre is None else 1.0,
        )
        withheld = blank or undetermined
        return _CoefRow(
            name="Intercept",
            coef=intercept,
            se=None if withheld else icpt_se,
            z=None if withheld else z,
            p=None if withheld else p,
            ci_low=None if withheld else ci_lo,
            ci_high=None if withheld else ci_hi,
            estimable=not undetermined,
            separated=blank,
        )

    # ── Producers: one per feature class ──────────────────────────

    def random_effect_rows(self, g: GroupSlice, spec: Any) -> list[_CoefRow]:
        return [self._structured_row(g, spec, "random_effect")]

    def factor_smooth_rows(self, g: GroupSlice, spec: Any) -> list[_CoefRow]:
        return [self._structured_row(g, spec, f"factor_smooth_{spec.basis}")]

    def _structured_row(self, g: GroupSlice, spec: Any, kind: str) -> _CoefRow:
        """One summary row for a structured term; its levels have a dedicated report."""
        active = g.name in self.selected
        lambdas = self.structured_lambdas(g.name)
        return _CoefRow(
            name=g.name,
            group=str(g.feature_name),
            structured_kind=kind,
            n_levels=len(spec._levels),
            n_params=g.size,
            active=active,
            group_norm=float(np.linalg.norm(self.beta[g.sl])) if active else 0.0,
            edf=self.group_edf().get(g.name, 0.0) if active else 0.0,
            smoothing_lambda=(lambdas[0][1] if len(lambdas) == 1 else None),
            smoothing_lambdas=lambdas,
        )

    def ordered_rows(self, g: GroupSlice, spec: Any) -> list[_CoefRow]:
        """An ordered term's rows, once for all its groups: the whole-term row(s), then its levels.

        A specials term owns a second, unpenalized GroupSlice under the same feature
        name. ``reconstruct`` needs the full-width vector, but every statistic on the
        whole-term row (edf, the test, ref_df, n_params) is a statement about the
        spline block.  ``feature_active`` (any block survived selection) gates the level
        table, as a free special keeps a real SE when the curve is dropped;
        ``smooth_active`` (the spline block alone) gates the whole-term row.
        """
        name = g.feature_name
        if name in self._ordered_done:
            return []
        self._ordered_done.add(name)
        feature_groups = [fg for fg in self.groups if fg.feature_name == name]
        smooth_groups = spline_groups(feature_groups)
        feature_active = any(fg.name in self.selected for fg in feature_groups)
        smooth_active = any(fg.name in self.selected for fg in smooth_groups)
        feature_edf = (
            sum(self.group_edf().get(fg.name, 0.0) for fg in smooth_groups)
            if smooth_active
            else 0.0
        )
        se_levels = self.feature_se(name)
        raw = spec.reconstruct(np.concatenate([self.beta[fg.sl] for fg in feature_groups]))
        inner_kind = spec.basis_kind
        if inner_kind == "spline":
            rows = [self._ordered_spline_row(g, spec, smooth_groups, smooth_active, feature_edf)]
        else:
            rows = self._ordered_parametric_rows(
                g, spec, inner_kind, smooth_groups[0], smooth_active, feature_edf
            )
        return rows + self._ordered_level_rows(g, spec, raw, se_levels, feature_active, inner_kind)

    def _ordered_spline_row(self, g, spec, smooth_groups, smooth_active, feature_edf) -> _CoefRow:
        """An ordered spline's Wood test row, over the active blocks of its smooth."""
        blocks = [fg for fg in smooth_groups if fg.name in self.active_by_name]
        size = sum(fg.size for fg in smooth_groups)
        test = self.untested(size)
        curve_se = (float("nan"), float("nan"))
        beta_active = (
            np.concatenate([self.beta[fg.sl] for fg in blocks])
            if blocks
            else np.empty(0, dtype=float)
        )
        if blocks:
            if not self.is_sandwich:
                active = [self.active_by_name[fg.name] for fg in blocks]
                columns = np.concatenate([np.arange(ag.start, ag.end) for ag in active])
                covariance = self.cov_from_aug(
                    covariance_selected_block(self.XtWX_inv_aug, columns + 1)
                )
                test = self.wood_test(
                    blocks,
                    beta_active,
                    columns,
                    covariance,
                    size,
                    catch=np.linalg.LinAlgError,
                )
            curve_se = self.curve_se_range(g.feature_name)
        fields = self.spline_fields(smooth_groups[0].name, spec._basis_spline)
        fields["edf"] = feature_edf
        return _CoefRow(
            name=str(g.feature_name),
            group=str(g.feature_name),
            is_spline=True,
            n_params=size,
            active=smooth_active,
            group_norm=float(np.linalg.norm(beta_active)),
            wald_chi2=test.stat if smooth_active else None,
            wald_p=test.p if smooth_active else None,
            ref_df=test.ref_df if smooth_active else None,
            curve_se_min=curve_se[0],
            curve_se_max=curve_se[1],
            subgroup_type="ordered_spline",
            estimable=not test.undetermined,
            # The wrapper forwards its basis's rules, and raises on a step-mode pickle, which is
            # safe because ``spec.reconstruct(...)`` and ``spec.basis_kind`` already raised the
            # migration error first.
            monotone=rules_of(spec).describe() or None,
            monotone_engine=self.engine_tag(spec, g.monotone_engine),
            monotone_repaired=g.feature_name in self.repaired,
            **fields,
        )

    def _ordered_parametric_rows(
        self, g, spec, inner_kind, main_group, smooth_active, feature_edf
    ) -> list[_CoefRow]:
        """An ordered Piecewise or Polynomial basis: a Wald row, then per-power or structural rows.

        The inner block is UNPENALIZED, so the whole-term test is a plain Wald chi-square,
        not Wood's smooth test: no smoothing parameter was estimated.
        """
        name, label = g.feature_name, str(g.feature_name)
        b_main = self.beta[main_group.sl]
        test = self.untested(main_group.size)
        V_b_main: NDArray | None = None
        determines = None
        active_group = self.active_by_name.get(main_group.name) if smooth_active else None
        if active_group is not None:
            V_b_main = self.group_block(active_group)
            determines = self.block_determines(main_group)
            test = self.block_wald(main_group, V_b_main)
        rows = [
            _CoefRow(
                name=label,
                group=label,
                is_spline=True,
                n_params=main_group.size,
                active=smooth_active,
                group_norm=float(np.linalg.norm(b_main)) if smooth_active else 0.0,
                wald_chi2=test.stat if smooth_active else None,
                wald_p=test.p if smooth_active else None,
                ref_df=test.ref_df if smooth_active else None,
                subgroup_type=f"ordered_{inner_kind}",
                estimable=not test.undetermined,
                edf=feature_edf,
            )
        ]
        if inner_kind == "polynomial":
            # Per-power rows, labelled by the stated powers as a numeric Polynomial's are.
            powers = spec._basis_spline.powers
            rows += [
                self.coefficient_row(main_group, i, f"{name}[P{powers[i]}]", label)
                for i in range(main_group.size)
            ]
        elif V_b_main is not None:
            # Structural contrast rows (Smith 1979): no per-segment per-power z rows,
            # which do not exist under C0 seams.
            rows += _ordered_piecewise_structural_rows(
                spec,
                name,
                label,
                b_main,
                V_b_main,
                self.alpha,
                is_sandwich=self.is_sandwich,
                sandwich_penalized=main_group.name in self.sandwich_penalized,
                wald_df_limit=self.wald_df_limit,
                determines=determines,
            )
        return rows

    def _ordered_level_rows(
        self, g, spec, raw, se_levels, feature_active, inner_kind
    ) -> list[_CoefRow]:
        """One row per ordered level, against the base level, with its Fit column word.

        Specials come from ``raw["special_levels"]``, not ``spec._specials``, which is
        string-coerced: for ``specials=[9]`` the level is 9 here and "9" there.  A level
        carried by an unpenalized parametric block is "piecewise"/"polynomial", not
        "smooth", beside its "free" neighbours.
        """
        special_labels = set(raw.get("special_levels") or ()) if spec.has_specials else None
        pinned_labels = set(raw.get("pinned_specials") or ())
        main_fit = "smooth" if inner_kind == "spline" else inner_kind
        rows = []
        for i, level in enumerate(raw["levels"]):
            coef = float(raw["level_log_relativities"][level])
            se: float | None = (
                float(se_levels[i]) if feature_active and i < len(se_levels) else None
            )
            estimable = se is None or not np.isnan(se)
            ci_lo: float | None
            ci_hi: float | None
            if se is not None and np.isfinite(se) and se > 0.0:
                _, _, ci_lo, ci_hi = _compute_coef_stats(coef, se, self.alpha)
            elif se is not None and np.isfinite(se) and level == spec._base_level:
                ci_lo = ci_hi = coef
            else:
                se = ci_lo = ci_hi = None
            fit = ordered_level_fit(
                level, special_labels=special_labels, pinned_labels=pinned_labels, main_fit=main_fit
            )
            rows.append(
                _CoefRow(
                    name=f"{g.feature_name}[{level}]",
                    group=str(g.feature_name),
                    coef=coef,
                    se=se,
                    ci_low=ci_lo,
                    ci_high=ci_hi,
                    estimable=estimable,
                    level_fit=fit,
                )
            )
        return rows

    def spline_rows(self, g: GroupSlice, spec: Any) -> list[_CoefRow]:
        """A spline group's Wood test row (a legacy ``linear`` subgroup's is a Wald test)."""
        fields = {
            **self.spline_fields(g.name, spec),
            **self.rule_fields(spec, g),
            "subgroup_type": g.subgroup_type,
        }
        if g.name not in self.selected:
            return [self.group_row(g, None, **{**fields, "edf": 0.0})]
        if g.subgroup_type == "linear":
            test = self.group_wald_test(g)
        else:
            test = self.group_wood_test(g)
        curve_min, curve_max = self.curve_se_range(g.feature_name)
        return [self.group_row(g, test, curve_se_min=curve_min, curve_se_max=curve_max, **fields)]

    def categorical_rows(self, g: GroupSlice, spec: Any) -> list[_CoefRow]:
        """One row per non-base level, with its experience and a separated level blank."""
        edf = self.group_edf().get(g.name, 0.0) if g.name in self.selected else 0.0
        diag = self.level_diag.get(g.name, {})
        rules = rules_of(spec).describe() or None
        engine = self.engine_tag(spec, g.monotone_engine if rules else None)
        rows = []
        for i, level in enumerate(spec._non_base):
            blank = self.moves(g.start + i)
            n_obs, exposure_share, thin = diag.get(i, (None, None, False))
            # a blank row's interval is not shown, so no note about it is either
            thin = thin and not blank
            rows.append(
                self.coefficient_row(
                    g,
                    i,
                    f"{g.name}[{level}]",
                    g.name,
                    blank=blank,
                    edf=edf if i == 0 else None,
                    level_n_obs=n_obs,
                    level_exposure_share=exposure_share,
                    quasi_separated=thin,
                    advisory_trigger="thin_level" if thin else "",
                    monotone=rules,
                    monotone_engine=engine,
                )
            )
        return rows

    def spline_by_level_rows(self, g: GroupSlice, spec: Any) -> list[_CoefRow]:
        """A by-level curve's Wood test row."""
        if g.name not in self.selected:
            return [self.group_row(g, None)]
        test = self.group_wood_test(g)
        return [
            self.group_row(
                g,
                test,
                edf=self.mapped_edf(g.name),
                smoothing_lambda=_resolve_group_lambda(g.name, self.reml_lambdas, self.lambda2),
            )
        ]

    def polynomial_rows(self, g: GroupSlice, spec: Any) -> list[_CoefRow]:
        """One row per stated power: with powers={1, 2, 4} the third row is [P4]."""
        powers = spec.powers
        if powers == tuple(range(1, spec.degree + 1)):
            group = f"{g.name} P({spec.degree})"
        else:
            group = f"{g.name} P({','.join(str(p) for p in powers)})"
        return [self.coefficient_row(g, i, f"{g.name}[P{powers[i]}]", group) for i in range(g.size)]

    def polynomial_by_level_rows(self, g: GroupSlice, spec: Any) -> list[_CoefRow]:
        """A by-level polynomial's Wald test row."""
        if g.name not in self.selected:
            return [self.group_row(g, None)]
        return [self.group_row(g, self.group_wald_test(g))]

    def crossed_level_rows(self, g: GroupSlice, spec: Any) -> list[_CoefRow]:
        """One row per crossed cell, a separated cell blank."""
        return [
            self.coefficient_row(
                g, i, f"{g.name}[{lev1}:{lev2}]", g.name, blank=self.moves(g.start + i)
            )
            for i, (lev1, lev2) in enumerate(spec._pairs)
        ]

    def numeric_by_level_rows(self, g: GroupSlice, spec: Any) -> list[_CoefRow]:
        """One slope row per non-base level."""
        return [
            self.coefficient_row(g, i, f"{g.name}[{level}]", g.name)
            for i, level in enumerate(spec._non_base)
        ]

    def numeric_interaction_rows(self, g: GroupSlice, spec: Any) -> list[_CoefRow]:
        """Up to four coefficient rows, else one Wald test row."""
        if g.name not in self.selected:
            return [self.group_row(g, None)]
        if g.size <= 4:
            return [
                self.coefficient_row(g, i, f"{g.name}[{i}]" if g.size > 1 else g.name, g.name)
                for i in range(g.size)
            ]
        test = self.group_wald_test(g)
        return [
            self.group_row(
                g,
                test,
                edf=self.mapped_edf(g.name),
                smoothing_lambda=_resolve_group_lambda(g.name, self.reml_lambdas, self.lambda2),
            )
        ]

    def tensor_rows(self, g: GroupSlice, spec: Any) -> list[_CoefRow]:
        """A tensor's Wood test row."""
        lam = _resolve_group_lambda(g.name, self.reml_lambdas, self.lambda2)
        if g.name not in self.selected:
            return [self.group_row(g, None, edf=0.0, smoothing_lambda=lam)]
        test = self.group_wood_test(g)
        return [self.group_row(g, test, edf=self.mapped_edf(g.name), smoothing_lambda=lam)]

    def numeric_rows(self, g: GroupSlice, spec: Any) -> list[_CoefRow]:
        edf = self.group_edf().get(g.name, 0.0) if g.name in self.selected else 0.0
        estimable = bool(self.estimable[g.start])
        return [self.coefficient_row(g, 0, g.name, g.name, edf=edf, estimable=estimable)]

    def piecewise_rows(self, g: GroupSlice, spec: Any) -> list[_CoefRow]:
        """Per-knot rows, then the whole-term row carrying the edf and a J+1 df chi-square.

        The edf is on the whole-term row alone, or the summary's parametric and smooth
        buckets would both count it.
        """
        # Per-knot Wald rows, CIs and edf = J+1 are valid only when all four hold; otherwise the
        # per-coefficient p-values are withdrawn: (1) the term is unpenalized (an L1 slope-change
        # penalty is trend filtering, whose df is data-dependent); (2) the group carries no selection
        # shrinkage (enforced here: z/p/CI and the whole-term chi-square are withheld, estimates and
        # SEs stay); (3) the breakpoints are fixed inputs, not selected on the response (the
        # statistic then converges to a supremum of a nonstandard Gaussian process even though df
        # stays nominally J+1; quantile placement ``breaks=int`` looks only at x and is milder; kinks
        # read off a fitted Spline and refit on the same data are the offender); (4) the term is
        # unconstrained (an active monotone constraint makes df the data-dependent size of the face).
        active = g.name in self.selected
        shrunk = g.name in (self.selection_shrunk_group_names or ())
        knots = spec._knots
        edf = self.group_edf().get(g.name, 0.0) if active else 0.0
        rows = [
            self.coefficient_row(
                g, i, f"{g.name}[{float(knots[knot_index]):.10g}]", g.name, withheld=shrunk
            )
            for i, knot_index in enumerate(spec._non_base_indices)
        ]
        test = None
        if active:
            test = _GroupTest(None, None, None) if shrunk else self.group_wald_test(g)
        # ``subgroup_type`` names the row for what it is in both renderers, which
        # otherwise label every group-test row "spline".
        rows.append(self.group_row(g, test, subgroup_type="piecewise", edf=edf))
        return rows

    def default_rows(self, g: GroupSlice, spec: Any) -> list[_CoefRow]:
        estimable = bool(self.estimable[g.start])
        return [self.coefficient_row(g, 0, g.name, g.name, estimable=estimable)]


def _flag_outsized_errors(rows: list[_CoefRow]) -> None:
    """Flag a non-categorical coefficient whose error is far above the model's typical one.

    The low-credibility advisory has two disjoint triggers, neither a separation test
    (separation is a joint property of design and response, issue #239). Primary:
    categorical levels with too little experience (``_thin_levels``). This fallback, for
    rows without per-level counts, is SE-based with threshold ``max(50 * median_se,
    10.0)``, which is not scale-invariant (rescaling one predictor trips it) and is
    dominated by the absolute 10.0 when the median parametric SE is under 0.2 (issue #304).
    """
    parametric_ses = [
        r.se
        for r in rows
        if r.se is not None
        and r.se > 0
        and r.p is not None
        and not r.is_spline
        and r.name != "Intercept"
    ]
    if not parametric_ses:
        return
    threshold = max(float(np.median(parametric_ses)) * 50, 10.0)
    for r in rows:
        if r.quasi_separated or r.is_spline or r.name == "Intercept" or r.p is None:
            continue
        if r.level_n_obs is None and r.se is not None and r.se > threshold:
            r.quasi_separated = True
            r.advisory_trigger = "outsized_se"


def build_coef_rows(
    *,
    groups: list[GroupSlice],
    specs: dict,
    interaction_specs: dict,
    result: Any,
    X_a: NDArray | MetricsDesign,
    W: NDArray,
    XtWX_inv: NDArray,
    XtWX_inv_aug: NDArray,
    active_groups: list[GroupSlice],
    known_scale: bool,
    group_edf_map: dict | None,
    reml_lambdas: dict | None,
    lambda2: float | dict,
    n_obs: int,
    alpha: float = 0.05,
    monotone_repairs: dict | None = None,
    # Precomputed inference quantities (avoids recomputing QR/EDF)
    precomputed_R_a: NDArray | None = None,
    precomputed_edf: NDArray | None = None,
    precomputed_edf1: NDArray | None = None,
    precomputed_design_moments: tuple[NDArray, NDArray, NDArray] | None = None,
    test_bases: dict[str, NDArray] | None = None,
    coefficient_estimable_override: NDArray | None = None,
    selected_group_names: set[str] | None = None,
    group_matrices: list | None = None,
    sample_weights: NDArray | None = None,
    weight_semantics: str | None = None,
    recorded_likelihood_size: float | None = None,
    selection_shrunk_group_names: set[str] | None = None,
    rules_enforced: bool = True,
    sandwich_covariance: Any | None = None,
    contrast_estimable: Callable[[NDArray], bool] | None = None,
    information_underflowed: bool | None = None,
    intercept_centre: NDArray | None = None,
) -> list[_CoefRow]:
    """Build coefficient table rows for summary output.

    Standalone function that can be called from ``ModelMetrics._build_coef_rows``
    or from ``SuperGLM.summary()`` without a ``ModelMetrics`` instance.

    Parameters
    ----------
    XtWX_inv : (p_active, p_active) inverse used for EDF computation.
    XtWX_inv_aug : (p_active+1, p_active+1) augmented inverse including
        intercept row/column, used for SE computation.
    selection_shrunk_group_names : groups the fitted penalty's lambda1 term
        shrinks (``penalties.base.selection_shrunk_group_names``).  Rows whose
        Wald validity is conditional on being unshrunk -- today the Piecewise
        branch -- withhold z/p/CI and the whole-term chi-square for these.
    rules_enforced : False for a model last fitted by ``fit_path``, which fits every
        term without its constraint rows; the rule line of a term the build
        constrained then says its rules were not enforced.
    contrast_estimable : whether the decomposition that decided the coefficient mask
        determines a contrast over the fitted coefficients
        (``RankInfo.is_estimable``, which also answers for a term's fitted function); the
        fit's own rank record's when the mask is.  A contrast whose coefficients the mask
        all shows is determined without it; without it, any other is not.
    information_underflowed : whether float64 holds none of the likelihood's information
        on the rows the covariance was formed from (``PIRLSResult.information_underflowed``,
        the fit's own when omitted): the intercept then shows no error, as no coefficient does.
    intercept_centre : the public column mean, over the fitted coefficients, that the
        decomposition behind ``contrast_estimable`` was centred on.  The reported
        intercept is the centred one, which the fit always determines, less
        ``intercept_centre' beta``: ``contrast_estimable(contrast, intercept=1.0)``
        decides the two together.  Without it, only the ordered terms' base-level
        adjustment is decided.
    """
    # The arguments are the table's inputs, field for field.
    return _CoefTable(**locals()).rows()


def build_basis_detail(
    groups,
    specs,
    interaction_specs,
    result,
    XtWX_inv_aug,
    active_groups,
    known_scale,
    alpha=0.05,
    coefficient_estimable_override=None,
    selected_group_names=None,
    sandwich_covariance=None,
):
    """Build per-coefficient detail for active 1-D spline groups.

    Uses the same known_scale-aware covariance path as ``build_coef_rows``
    so that SE/z/p/CI values are consistent with the main summary.
    """
    from superglm.features.spline import _SplineBase

    is_sandwich = sandwich_covariance is not None
    penalized = frozenset(sandwich_covariance.penalized_group_names) if is_sandwich else frozenset()
    selected_names = _selected_names(result, groups, selected_group_names)
    coefficient_estimable = _coefficient_mask(result, coefficient_estimable_override)
    read_errors = _error_reader(result, known_scale, is_sandwich)
    active_by_name = {group.name: group for group in active_groups}
    detail: dict[str, list] = {}
    for g in groups:
        # V1: skip interactions
        if g.feature_name in interaction_specs or not isinstance(
            specs.get(g.feature_name), _SplineBase
        ):
            continue
        ag = active_by_name.get(g.name)
        if g.name not in selected_names or ag is None:
            continue
        se_arr = read_errors(augmented_group_diagonal(XtWX_inv_aug, ag)).astype(float, copy=True)
        se_arr[~coefficient_estimable[g.sl]] = np.nan
        withhold_zp = is_sandwich and g.name in penalized
        detail[g.name] = [
            _basis_detail_row(
                g, i, float(result.beta[g.start + i]), float(se_arr[i]), alpha, withhold_zp
            )
            for i in range(g.size)
        ]
    return detail


def _basis_detail_row(
    g: GroupSlice, i: int, coef: float, se: float, alpha: float, withhold_zp: bool
) -> _BasisDetailRow:
    z, p, ci_lo, ci_hi = _compute_coef_stats(coef, se, alpha)
    return _BasisDetailRow(
        parent_name=g.name,
        basis_index=i,
        coef=coef,
        se=se,
        z=None if withhold_zp else z,
        p=None if withhold_zp else p,
        ci_low=ci_lo,
        ci_high=ci_hi,
    )
