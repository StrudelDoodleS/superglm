"""What ``superglm.cross_validate`` reads of a SuperLSS.

Each fold is fitted on its training rows and scored on its held-out rows, with
any smoothing selection inside the fold. The built-in scores are the
negative log-likelihood and the CRPS, both proper (Gneiting and Raftery, 2007),
and the Gini and actual over expected of the conditional mean. A score is a
ratio of sums, so ``cross_validate`` pools it across folds the same way.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm._frame import as_eager_frame
from superglm.distributional.checks.scores import (
    _retained_crps,
    _retained_log_score,
    _scoring_rows,
)
from superglm.distributional.family import DistributionFunctionFamily
from superglm.distributional.offsets import (
    offset_columns,
    shape_fit_offsets,
    sources_of,
    sum_offset_sources,
    take_offset,
)
from superglm.distributional.weights import (
    UnsupportedLikelihoodContractError,
    resolve_likelihood_weights,
)
from superglm.model.binding_ops import bind_feature_templates
from superglm.validation import _normalized_gini

logger = logging.getLogger(__name__)


def _nll_parts(model, X, y, *, sample_weight=None, offset=None) -> tuple[float, float]:
    """Total negative log-likelihood and the rows' likelihood size."""
    fitted = model._require_fitted()
    rows = _scoring_rows(fitted, X, y, sample_weight=sample_weight, offsets=offset)
    return float(np.sum(_retained_log_score(fitted, rows))), float(rows.aggregation_mass.sum())


def _crps_parts(model, X, y, *, sample_weight=None, offset=None) -> tuple[float, float]:
    """Total CRPS and the rows' likelihood size.

    NaN for a family with no distribution function, or where the numeric route
    cannot resolve the integral, so the fold keeps its other scores. The mean CRPS
    then leaves that fold out, so the reason is logged. A prediction that fails is
    not one of these: it fails the fold.
    """
    fitted = model._require_fitted()
    rows = _scoring_rows(fitted, X, y, sample_weight=sample_weight, offsets=offset)
    mass = rows.aggregation_mass
    theta = np.asarray(
        fitted._predict_parameters_resolved(rows.frame, rows.offsets), dtype=np.float64
    )
    try:
        scores = _retained_crps(fitted, rows, method="auto", n_nodes=64, theta=theta)
    except (NotImplementedError, UnsupportedLikelihoodContractError, ValueError) as refusal:
        logger.warning("A fold has no CRPS, and the mean CRPS leaves it out: %s", refusal)
        return float("nan"), float(mass.sum())
    return float(np.sum(mass * scores)), float(mass.sum())


def _scored_means(
    model, X, y, sample_weight, offset
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
    """The response, conditional mean and weight of the rows with positive weight."""
    rows = _scoring_rows(model._require_fitted(), X, y, sample_weight=sample_weight, offsets=offset)
    weights = (
        None if sample_weight is None else np.asarray(sample_weight, np.float64)[rows.positions]
    )
    mean = np.asarray(
        model._require_fitted().family.default_prediction(
            model._require_fitted()._predict_parameters_resolved(rows.frame, rows.offsets)
        ),
        dtype=np.float64,
    )
    return rows.response, mean, weights


def _actual_expected_parts(model, X, y, *, sample_weight=None, offset=None) -> tuple[float, float]:
    """Weighted actual and weighted expected totals."""
    response, mean, weights = _scored_means(model, X, y, sample_weight, offset)
    if weights is None:
        return float(np.sum(response)), float(np.sum(mean))
    return float(np.sum(weights * response)), float(np.sum(weights * mean))


def _gini(model, X, y, *, sample_weight=None, offset=None) -> float:
    """Normalized Gini of the conditional mean; NaN where the rows' total is not positive.

    It measures how the mean orders a share of a positive total, so with none
    there is no such share to order, on a response that can be negative or a
    fold with no claims.
    """
    response, mean, weights = _scored_means(model, X, y, sample_weight, offset)
    total = np.sum(response) if weights is None else np.sum(weights * response)
    if total > 0.0:
        return _normalized_gini(response, mean, weights)
    logger.warning(
        "A fold's total response is not positive, so it has no Gini, and the mean Gini leaves it out."
    )
    return float("nan")


def parts_ratio(numerator: float, denominator: float) -> float:
    """A score from its parts; NaN where a zero denominator leaves it undefined."""
    return float(numerator / denominator) if denominator != 0.0 else float("nan")


def _parts_quotient(parts):
    def score(model, X, y, *, sample_weight=None, offset=None) -> float:
        return parts_ratio(*parts(model, X, y, sample_weight=sample_weight, offset=offset))

    return score


_POOLED_PARTS = {
    "nll": _nll_parts,
    "crps": _crps_parts,
    "actual_expected": _actual_expected_parts,
}
_SCORERS = {
    "nll": _parts_quotient(_nll_parts),
    "crps": _parts_quotient(_crps_parts),
    "gini": _gini,
    "actual_expected": _parts_quotient(_actual_expected_parts),
}


class LSSFolds:
    """How ``cross_validate`` clones, fits, reads and scores a SuperLSS's folds."""

    offset_argument = "offsets"
    warm_argument = "lambdas"
    # A Fellner-Schall search can take longer from another fold's smoothing than from
    # its own data-scaled start, and can reach its iteration limit.
    default_warm_start = False

    @staticmethod
    def default_fit_mode(fit_kwargs: Mapping[str, Any]) -> str:
        """``fit`` holds given smoothing fixed; with none given, each fold estimates its own."""
        return "fit" if "lambdas" in fit_kwargs else "fit_reml"

    def refuse_fit(self, fit_mode: str, fit_kwargs: Mapping[str, Any]) -> None:
        """Refuse a fixed-smoothing fit that names no smoothing for a fitted model's penalties."""
        if fit_mode != "fit" or "lambdas" in fit_kwargs or self.model._model is None:
            return
        missing = [
            component.name
            for component in self.model._model.layout.penalties
            if component.lambda_policy is None or component.lambda_policy.mode != "fixed"
        ]
        if missing:
            raise ValueError(
                f"fit_mode='fit' holds smoothing fixed, but {missing} have none set. Use "
                "fit_mode='fit_reml' to estimate it in each fold, or pass "
                "fit_kwargs={'lambdas': {...}}."
            )

    scorers = _SCORERS
    pooled_parts = _POOLED_PARTS
    # Bounds per row belong to a fit and to its scores; a fold's held-out scores have none.
    unsliced_fit_kwargs = frozenset({"truncation", "censored"})
    # A fold is read for its predictions and curves, not its training diagnostics.
    fold_fit_kwargs: Mapping[str, Any] = {"retain_rows": False}

    row_cost_label = "Log score"

    drops_zero_weight_rows = True

    def __init__(self, model) -> None:
        self.model = model

    @property
    def default_scoring(self) -> tuple[str, ...]:
        """Every built-in score the family has: a CRPS needs its distribution function."""
        return tuple(
            name
            for name in _SCORERS
            if name != "crps" or isinstance(self.model.family, DistributionFunctionFamily)
        )

    def read_columns(self) -> tuple[str, ...]:
        """The columns the predictors read: their terms and their interactions' parents."""
        names = [
            name
            for predictor in self.model._predictors
            for name in (
                *predictor.features,
                *offset_columns((predictor,)),
                *(
                    parent
                    for spec in predictor.interaction_specs.values()
                    for parent in spec.parent_names
                ),
            )
        ]
        return tuple(sorted(dict.fromkeys(names), key=repr))

    def weights_must_be_positive(self) -> bool:
        """The fit resolves zero weights itself, by dropping those rows."""
        return False

    def prepare_offset(self, offset, frame, sample_weight):
        """Held-out replacements for declared arrays; fit sources stay on templates."""
        if offset is not None:
            raise ValueError(
                "Declare SuperLSS offsets on predictor helpers with offset=, not cross_validate"
            )
        shaped = shape_fit_offsets(self.model._predictors, frame)
        arrays = {
            p.name: tuple(
                v
                for s, v in zip(sources_of(p), shaped[p.name], strict=True)
                if not isinstance(s, str)
            )
            for p in self.model._predictors
            if any(not isinstance(s, str) for s in sources_of(p))
        }
        resolved = resolve_likelihood_weights(
            sample_weight, n_observations=len(frame), contract=self.model._weight_contract
        )
        kept = sum_offset_sources(arrays, resolved.input_positions)
        replacements = {}
        for name, values in kept.items():
            replacements[name] = np.zeros(len(frame), dtype=np.float64)
            replacements[name][resolved.input_positions] = values
        return replacements or None

    def fit_offset_kwargs(self, offset) -> dict:
        """The public fit receives no offsets; its declarations have been sliced."""
        return {}

    def level_bindings(self, frame, sample_weight) -> dict[str, dict[Any, Any]]:
        """Each predictor's level universes and bases on the rows the fit keeps.

        A SuperLSS fit drops a row of weight zero before it reads anything, so the
        levels are read from the retained rows alone, under the weights the fit
        reads them under: physical rows under prior weights, the counts under
        frequency weights.
        """
        resolved = resolve_likelihood_weights(
            sample_weight, n_observations=len(frame), contract=self.model._weight_contract
        )
        if resolved.provenance.dropped_count:
            frame = as_eager_frame(frame.take_rows(resolved.input_positions))
        return {
            predictor.name: bind_feature_templates(
                tuple(predictor.features.items()), frame, resolved.geometry_values
            )
            for predictor in self.model._predictors
        }

    def estimator(self, bindings: Mapping[str, Mapping[Any, Any]], *, rows=None):
        """A fresh unfitted copy with the shared level universes; the model is not changed."""
        clone = self.model._clone_with_replaced_terms({})
        if rows is not None:
            clone._predictors = tuple(
                replace(p, offset=take_offset(getattr(p, "offset", None), rows))
                for p in clone._predictors
            )
        for predictor in clone._predictors:
            for name, binding in bindings[predictor.name].items():
                predictor.features[name].apply_level_binding(binding)
        return clone

    def fit_summary(self, estimator, fit_mode: str) -> dict[str, Any]:
        result = estimator.result_
        summary = {
            "converged": bool(result.converged),
            "n_iter": result.n_inner_iter,
            "effective_df": result.total_effective_df,
        }
        if fit_mode == "fit_reml":
            summary["n_reml_iter"] = int(result.n_smoothing_iter)
        return summary

    def live_lambdas(self, estimator) -> dict[str, float]:
        """The fit's smoothing parameters, less each term with a component on its last stop.

        A search started at the infinity face or the cap cannot reach another
        fold's optimum, and a term's penalties start cold together.
        """
        stuck = {
            *estimator.exact_face_components_,
            *(estimator.smoothing_unresolved_upper_bound_ or ()),
        }
        cold_terms = {name.rpartition("#")[0] for name in stuck}
        return {
            name: float(value)
            for name, value in estimator.smoothing_parameters_.items()
            if name.rpartition("#")[0] not in cold_terms
        }

    def row_costs(self, estimator, X, y, sample_weight, offset) -> NDArray[np.float64]:
        """Each row's log score; a row of zero weight scores nothing."""
        fitted = estimator._require_fitted()
        rows = _scoring_rows(fitted, X, y, sample_weight=sample_weight, offsets=offset)
        costs = np.zeros(rows.n_observations, dtype=np.float64)
        costs[rows.positions] = _retained_log_score(fitted, rows)
        return costs

    def predict(self, estimator, X, offset) -> NDArray[np.float64]:
        return estimator.predict(X, offsets=offset)

    def curve_similarity(self, estimators, X, sample_weight) -> None:
        """None: the editor's Cross-validation tab draws a SuperLSS's fold curves."""
        del estimators, X, sample_weight
