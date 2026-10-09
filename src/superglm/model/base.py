"""Constructor, prediction, and core helpers for SuperGLM."""

from __future__ import annotations

import copy
import logging
import math
import warnings
from collections.abc import Hashable, Mapping
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from superglm._frame import EagerFrame, FrameLike, as_eager_frame
from superglm.distributions import Distribution, clip_mu
from superglm.dm_builder import (
    add_interaction,
    auto_detect_features,
    build_design_matrix,
    rebuild_design_matrix_with_lambdas,
    resolve_discrete_n_bins,
    should_discretize,
    should_discretize_tensor_interaction,
    validate_term_name_namespace,
)
from superglm.features.ordered_categorical import resolve_interaction_parent_of
from superglm.group_matrix import DesignMatrix, _discretize_column
from superglm.links import Link, stabilize_eta
from superglm.model.fit_state import (
    configured_family,
    configured_lambda2,
    configured_link,
    configured_penalty,
    fitted_lambda2,
    fitted_penalty,
)
from superglm.model.input_validation import validate_prediction_offset, validate_x_columns
from superglm.penalties.base import (
    Penalty,
    penalty_has_targets,
    penalty_targets_group,
    validate_penalty_features,
)
from superglm.penalties.group_elastic_net import GroupElasticNet
from superglm.penalties.group_lasso import GroupLasso
from superglm.penalties.ridge import Ridge
from superglm.penalties.sparse_group_lasso import SparseGroupLasso
from superglm.solvers.dispersion import model_weight_semantics, validate_weight_semantics
from superglm.solvers.pirls import PIRLSResult
from superglm.types import FeatureSpec, FitStats, GroupSlice

logger = logging.getLogger(__name__)

SelectionPenalty = float | Literal["auto"] | None

_SELECTION_PENALTY_ERROR = "selection_penalty must be None, 'auto', or a finite non-negative number"

_PENALTY_SHORTCUTS: dict[str, type[Any]] = {
    "group_lasso": GroupLasso,
    "group_elastic_net": GroupElasticNet,
    "sparse_group_lasso": SparseGroupLasso,
    "ridge": Ridge,
}


def _group_beta_indices(groups: list[GroupSlice], feature_name: object) -> NDArray[np.intp]:
    """Concatenate coefficient indices for one feature or interaction."""
    idx = [
        np.arange(g.start, g.end, dtype=np.intp) for g in groups if g.feature_name == feature_name
    ]
    if not idx:
        raise KeyError(f"No fitted groups found for feature {feature_name!r}")
    if len(idx) == 1:
        return idx[0]
    return np.concatenate(idx)


def _validate_group_feature_names(model, groups: list[GroupSlice]) -> None:
    """Reject incomplete design bookkeeping before any coefficient solve."""
    expected = (*model._feature_order, *model._interaction_order)
    missing = [name for name in expected if not any(group.feature_name == name for group in groups)]
    if missing:
        raise ValueError(f"No design-matrix groups were built for configured features: {missing}")


def _fit_discretizer_metadata(values: NDArray, n_bins: int) -> dict[str, Any]:
    """Compile fit-time support metadata for a fast discrete predictor."""
    support, _ = _discretize_column(values, n_bins)
    unique_vals = np.unique(values)
    if len(unique_vals) <= n_bins:
        return {
            "mode": "exact_support",
            "support": support,
        }
    return {
        "mode": "uniform_bins",
        "support": support,
        "lo": float(np.min(values)),
        "hi": float(np.max(values)),
        "n_bins": len(support),
    }


def _discretize_against_fit_metadata(
    values: NDArray,
    metadata: dict[str, Any],
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Discretize prediction data against fit-time support metadata."""
    values = np.asarray(values, dtype=np.float64).ravel()
    support = np.asarray(metadata["support"], dtype=np.float64)
    if metadata["mode"] == "exact_support":
        if len(support) <= 1:
            return support, np.zeros(len(values), dtype=np.intp)
        boundaries = 0.5 * (support[:-1] + support[1:])
        bin_idx = np.searchsorted(boundaries, values, side="right").astype(np.intp)
        return support, bin_idx

    lo = float(metadata["lo"])
    hi = float(metadata["hi"])
    n_bins = int(metadata["n_bins"])
    if lo == hi:
        return np.array([lo], dtype=np.float64), np.zeros(len(values), dtype=np.intp)
    edges = np.linspace(lo, hi, n_bins + 1)
    bin_idx = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, n_bins - 1)
    return support, np.asarray(bin_idx, dtype=np.intp)


def _compile_feature_fast_discrete_metadata(
    model,
    name: str,
    spec: FeatureSpec,
    frame: EagerFrame | None,
) -> dict[str, Any] | None:
    """Compile fit-time metadata for a discretized main-effect fast predictor."""
    if not should_discretize(spec, model._discrete):
        return None
    if frame is None:
        return None
    n_bins = resolve_discrete_n_bins(name, spec, model._n_bins)
    return {
        "kind": "feature",
        "discretizer": _fit_discretizer_metadata(
            frame.column_array(name, dtype=np.float64),
            n_bins,
        ),
    }


def _compile_interaction_fast_discrete_metadata(
    model,
    spec: Any,
    frame: EagerFrame | None,
) -> dict[str, Any] | None:
    """Compile fit-time metadata for a discretized tensor fast predictor."""
    if not should_discretize_tensor_interaction(spec, model._specs, model._discrete):
        return None
    if frame is None:
        return None
    left_name, right_name = spec.parent_names
    left_bins = resolve_discrete_n_bins(left_name, model._specs[left_name], model._n_bins)
    right_bins = resolve_discrete_n_bins(right_name, model._specs[right_name], model._n_bins)
    return {
        "kind": "interaction",
        "left_discretizer": _fit_discretizer_metadata(
            frame.column_array(left_name, dtype=np.float64),
            left_bins,
        ),
        "right_discretizer": _fit_discretizer_metadata(
            frame.column_array(right_name, dtype=np.float64),
            right_bins,
        ),
        "left_marginal": copy.deepcopy(spec._marginal1),
        "right_marginal": copy.deepcopy(spec._marginal2),
        "r_inv": None if getattr(spec, "_R_inv", None) is None else np.asarray(spec._R_inv).copy(),
    }


def _compile_fast_prediction_state(model) -> dict[str, dict[str, dict[str, Any] | None]]:
    """Freeze fit-time fast prediction metadata on the model."""
    needs_fit_frame = any(
        should_discretize(model._specs[name], model._discrete) for name in model._feature_order
    ) or any(
        should_discretize_tensor_interaction(
            model._interaction_specs[name],
            model._specs,
            model._discrete,
        )
        for name in model._interaction_order
    )
    fit_frame = (
        as_eager_frame(model._fit_X_ref)
        if needs_fit_frame and model._fit_X_ref is not None
        else None
    )
    return {
        "features": {
            name: _compile_feature_fast_discrete_metadata(
                model,
                name,
                model._specs[name],
                fit_frame,
            )
            for name in model._feature_order
        },
        "interactions": {
            name: _compile_interaction_fast_discrete_metadata(
                model,
                model._interaction_specs[name],
                fit_frame,
            )
            for name in model._interaction_order
        },
    }


def _build_prediction_plan(
    model,
    *,
    fast_prediction_state: dict[str, dict[str, dict[str, Any] | None]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Compile reusable metadata for prediction scoring."""
    if fast_prediction_state is None:
        fast_prediction_state = getattr(model, "_fast_prediction_state", None)
    if fast_prediction_state is None:
        fast_prediction_state = _compile_fast_prediction_state(model)
    return {
        "features": [
            {
                "kind": "feature",
                "name": name,
                "spec": model._specs[name],
                "beta_idx": _group_beta_indices(model._groups, name),
                "fast_discrete": copy.deepcopy(fast_prediction_state["features"].get(name)),
            }
            for name in model._feature_order
        ],
        "interactions": [
            {
                "kind": "interaction",
                "name": name,
                "spec": model._interaction_specs[name],
                "parent_names": tuple(model._interaction_specs[name].parent_names),
                "parent_specs": tuple(
                    model._specs.get(p) for p in model._interaction_specs[name].parent_names
                ),
                "beta_idx": _group_beta_indices(model._groups, name),
                "fast_discrete": copy.deepcopy(fast_prediction_state["interactions"].get(name)),
            }
            for name in model._interaction_order
        ],
    }


def _score_feature(spec, values: NDArray, beta: NDArray) -> NDArray[np.floating]:
    """Score a main-effect contribution on new data."""
    if hasattr(spec, "score"):
        return np.asarray(spec.score(values, beta), dtype=np.float64).ravel()
    return np.asarray(spec.transform(values) @ beta, dtype=np.float64).ravel()


def _score_interaction(spec, left: NDArray, right: NDArray, beta: NDArray) -> NDArray[np.floating]:
    """Score an interaction contribution on new data."""
    if hasattr(spec, "score"):
        return np.asarray(spec.score(left, right, beta), dtype=np.float64).ravel()
    return np.asarray(spec.transform(left, right) @ beta, dtype=np.float64).ravel()


def _prediction_plan(model) -> dict[str, list[dict[str, Any]]]:
    """Return the cached prediction metadata, building it lazily."""
    plan = model._prediction_plan
    if plan is None:
        fast_prediction_state = getattr(model, "_fast_prediction_state", None)
        if fast_prediction_state is None:
            fast_prediction_state = _compile_fast_prediction_state(model)
            model._fast_prediction_state = fast_prediction_state
        plan = _build_prediction_plan(model, fast_prediction_state=fast_prediction_state)
        model._prediction_plan = plan
    return plan


def freeze_prediction_plan(model) -> None:
    """Freeze the fast discrete prediction metadata after fitting."""
    fast_prediction_state = _compile_fast_prediction_state(model)
    model._fast_prediction_state = fast_prediction_state
    model._prediction_plan = _build_prediction_plan(
        model,
        fast_prediction_state=fast_prediction_state,
    )


def _score_feature_fast_discrete(
    term: dict[str, Any],
    X: EagerFrame,
    beta: NDArray,
) -> NDArray[np.floating]:
    """Approximate one canonical main-effect term via discretized support points."""
    support, bin_idx = _discretize_against_fit_metadata(
        X.column_array(term["name"], dtype=np.float64),
        term["fast_discrete"]["discretizer"],
    )
    values = _score_feature(term["spec"], support, beta)
    return np.asarray(values, dtype=np.float64).ravel()[bin_idx]


def _score_interaction_fast_discrete(
    model,
    term: dict[str, Any],
    X: EagerFrame,
    beta: NDArray,
) -> NDArray[np.floating]:
    """Approximate one canonical tensor term via discretized support pairs."""
    spec = term["spec"]
    metadata = term["fast_discrete"]
    left_name, right_name = term["parent_names"]
    left_support, idx1 = _discretize_against_fit_metadata(
        X.column_array(left_name, dtype=np.float64),
        metadata["left_discretizer"],
    )
    right_support, idx2 = _discretize_against_fit_metadata(
        X.column_array(right_name, dtype=np.float64),
        metadata["right_discretizer"],
    )
    B1_unique = np.asarray(
        spec._centered_marginal_basis(left_support, metadata["left_marginal"]).toarray(),
        dtype=np.float64,
    )
    B2_unique = np.asarray(
        spec._centered_marginal_basis(right_support, metadata["right_marginal"]).toarray(),
        dtype=np.float64,
    )

    n_support2 = len(right_support)
    pair_codes = idx1.astype(np.int64) * n_support2 + idx2.astype(np.int64)
    observed_codes, pair_idx = np.unique(pair_codes, return_inverse=True)
    observed_i1 = (observed_codes // n_support2).astype(np.intp)
    observed_i2 = (observed_codes % n_support2).astype(np.intp)
    B_joint = np.einsum(
        "ij,ik->ijk",
        B1_unique[observed_i1],
        B2_unique[observed_i2],
        optimize=True,
    ).reshape(len(observed_codes), -1)

    beta_block = beta
    r_inv = metadata["r_inv"]
    if r_inv is not None:
        beta_block = np.asarray(r_inv, dtype=np.float64) @ beta
    support_values = np.asarray(B_joint @ beta_block, dtype=np.float64).ravel()
    return support_values[np.asarray(pair_idx, dtype=np.intp)]


def _score_prediction_term_local_exact(
    term: dict[str, Any],
    X: EagerFrame,
    beta: NDArray,
) -> NDArray[np.floating]:
    """Score one canonical term from its term-local coefficient vector."""
    beta = np.asarray(beta, dtype=np.float64).ravel()
    expected_width = len(term["beta_idx"])
    if beta.shape != (expected_width,):
        raise ValueError(
            f"term {term['name']!r} requires {expected_width} coefficients, got {len(beta)}"
        )
    if term["kind"] == "feature":
        return _score_feature(term["spec"], X.column_array(term["name"]), beta)

    left_name, right_name = term["parent_names"]
    # A plan cached before parent specs were stashed carries no "parent_specs"
    # key; ``None`` then resolves through the identity path.  Resolution is
    # keyed on the INTERACTION too: a FactorSmooth reads its second parent as a
    # grouping column, so its columns must arrive exactly as the fit factorized
    # them (see ``resolve_interaction_parent_of``).
    left_spec, right_spec = term.get("parent_specs", (None, None))
    ispec = term["spec"]
    _, left = resolve_interaction_parent_of(ispec, left_spec, X.column_array(left_name))
    _, right = resolve_interaction_parent_of(ispec, right_spec, X.column_array(right_name))
    return _score_interaction(ispec, left, right, beta)


def _score_prediction_term_exact(
    term: dict[str, Any],
    X: EagerFrame,
    beta_all: NDArray,
) -> NDArray[np.floating]:
    """Score one canonical term exactly on the requested rows."""
    return _score_prediction_term_local_exact(
        term,
        X,
        beta_all[term["beta_idx"]],
    )


# ``model.predict`` -> ``_predict_exact`` -> ``predict_exact`` -> ``predict_eta_exact``
# -> ``_predict_eta``: a prediction warning names the caller of ``predict``.
_PREDICTION_WARNING_STACKLEVEL = 6


def _score_unidentified_factor_smooth(
    term: dict[str, Any],
    X: EagerFrame,
    beta_all: NDArray,
    *,
    population: bool,
) -> tuple[NDArray[np.floating], tuple]:
    """An ``sz`` term with levels left out of its population, as predicted (#432, ``_score_identified``)."""
    left_name, right_name = term["parent_names"]
    left_spec, right_spec = term.get("parent_specs", (None, None))
    spec = term["spec"]
    _, left = resolve_interaction_parent_of(spec, left_spec, X.column_array(left_name))
    _, right = resolve_interaction_parent_of(spec, right_spec, X.column_array(right_name))
    beta = np.asarray(beta_all[term["beta_idx"]], dtype=np.float64)
    return spec._score_identified(left, right, beta, population=population)


def prediction_centred_state(result) -> tuple[float, NDArray | None, float | None]:
    """The intercept, column centre and intercept remainder a public result's predictor starts from.

    ``(centred_intercept, state_center, centred_intercept_lo)`` when the
    result carries its fit's centred state (one-engine design §3.8; read into
    the public coordinates by ``runtime_canonicalize._public_centred_state``),
    the remainder ``None`` unless the fit published a compensated intercept
    (``mode_score.centred_intercept_remainder``), else the raw ``(intercept,
    None, None)``: a model saved before the state existed predicts as before.
    A revision of the coefficients carries the state with it
    (``fit_state.publish_revised_coefficients``).
    """
    alpha = getattr(result, "centred_intercept", None)
    centre = getattr(result, "state_center", None)
    if alpha is None or centre is None:
        return float(result.intercept), None, None
    alpha_lo = getattr(result, "centred_intercept_lo", None)
    return (
        float(alpha),
        np.asarray(centre, dtype=np.float64),
        None if alpha_lo is None else float(alpha_lo),
    )


def start_eta(n: int, intercept: float, intercept_lo: float | None) -> NDArray[np.float64]:
    """The accumulator a predictor's term contributions are added to.

    The intercept itself, or for a compensated pair its remainder: the terms
    and ``alpha_lo`` add at their own scale and ``finish_eta`` adds ``alpha``
    once, ``alpha + (sum_t score_t + alpha_lo)`` as ``linear_predictor``
    evaluates the fit.
    """
    return np.full(n, intercept if intercept_lo is None else intercept_lo, dtype=np.float64)


def finish_eta(eta: NDArray, intercept: float, intercept_lo: float | None) -> NDArray:
    """Close ``start_eta``'s accumulator: add a compensated pair's ``alpha`` last."""
    return eta if intercept_lo is None else intercept + eta


class EtaSum:
    """A public predictor's sum of its intercept and its term contributions.

    As fitted, ``start_eta`` and ``finish_eta``: ``alpha + (alpha_lo + sum_t
    score_t)``, bit for bit.  When a revision carried a centred column's change
    into the pair (``PIRLSResult.centred_sum_compensated``), ``alpha`` holds
    that column's ``c dbeta`` and can cancel against a row's ``(x - c) beta``.
    The sum is then compensated (``mode_score.CompensatedSum``), with ``alpha``
    first, each centred column as its exact pieces (``_add_centred_term``) and
    ``alpha_lo`` joining the errors last, in row chunks.  A row the expansion
    cannot finish takes the plain predictor (``CompensatedSum``).
    ``mode_score.linear_predictor`` evaluates the solver's pair the same way.

    ``add`` takes a term's rows or a callable giving them, and returns the
    handle ``finish(without=...)`` drops: the rows as added, or the
    compensated sum's addend, whose callable recomputes the rows if a row's
    fallback needs them, so the sum holds no term's rows.
    """

    __slots__ = ("compensated", "intercept", "intercept_lo", "total")

    def __init__(
        self, n: int, intercept: float, intercept_lo: float | None, compensated: bool = False
    ) -> None:
        from superglm.solvers.mode_score import CompensatedSum

        self.intercept, self.intercept_lo = intercept, intercept_lo
        self.compensated = bool(compensated)
        self.total: NDArray | CompensatedSum = (
            CompensatedSum(float(intercept), float(intercept_lo or 0.0), n=n)
            if self.compensated
            else start_eta(n, intercept, intercept_lo)
        )

    def add(self, values):
        """Add a term's rows (or a callable giving them); the handle a drop removes."""
        if self.compensated:
            return self.total.add(values)
        values = values() if callable(values) else values
        self.total += values
        return values

    def __iadd__(self, values) -> EtaSum:
        self.add(values)
        return self

    def finish(self, without=(), consume: bool = False) -> NDArray:
        """The predictor, less the terms whose handles are ``without`` (a drop).

        ``consume`` lets a compensated sum write the predictor over its own
        running total, which ends the sum.
        """
        if not self.compensated:
            total = self.total
            for contribution in without:
                total = total - contribution
            return finish_eta(total, self.intercept, self.intercept_lo)
        if not without:
            return self.total.value(consume=consume)
        total = self.total.copy()
        for addend in without:
            total.subtract(addend)
        return total.value(consume=True)


def scores_centred(spec) -> bool:
    """Whether a term's spec scores its raw columns about a centre (``_score_centred``).

    Only a raw numeric column (and the product of two) can sit at an offset
    far above its spread; every other column type is bounded by its basis, so
    its centre is folded into the intercept instead and it is scored as it is.
    """
    return callable(getattr(spec, "_score_centred", None))


def _centred_term_contribution(
    term: dict[str, Any],
    X: EagerFrame,
    beta_all: NDArray,
    centre: NDArray,
) -> NDArray[np.floating]:
    """``(B - 1 c') beta`` for a term whose fitted columns carry a centre.

    A raw numeric column is differenced before its product
    (``_score_centred``), so a column at a large offset contributes rows of
    its spread rounded at ``|x - c| |beta|``, as the fit's
    ``mode_score.centred_matvec`` formed them.  Any other term (which the
    published state never centres) is scored and its ``c' beta`` subtracted,
    the same algebra.
    """
    beta = np.asarray(beta_all[term["beta_idx"]], dtype=np.float64)
    spec = term["spec"]
    columns = _term_columns(term, X)
    if scores_centred(spec):
        return np.asarray(spec._score_centred(*columns, beta, centre), dtype=np.float64).ravel()
    scored = np.asarray(spec.score(*columns, beta), dtype=np.float64).ravel()
    return scored - math.fsum(centre * beta)


def _term_columns(term: dict[str, Any], X: EagerFrame) -> tuple[NDArray, ...]:
    """The raw columns a term is scored on: its own, or its two parents' as fitted."""
    if term["kind"] == "feature":
        return (X.column_array(term["name"]),)
    spec = term["spec"]
    left_name, right_name = term["parent_names"]
    left_spec, right_spec = term.get("parent_specs", (None, None))
    _, left = resolve_interaction_parent_of(spec, left_spec, X.column_array(left_name))
    _, right = resolve_interaction_parent_of(spec, right_spec, X.column_array(right_name))
    return (left, right)


def _add_centred_term(
    eta: EtaSum,
    term: dict[str, Any],
    X: EagerFrame,
    beta_all: NDArray,
    centre: NDArray,
):
    """Add ``(B - 1 c') beta`` to a compensated ``EtaSum``; the handle a drop removes.

    A column scored about its centre is added as
    ``mode_score.centred_column_expansion`` of its values: pieces exact to
    ``u^2`` and its fitted ``(x - c) beta`` bit for bit.  Any other term is its
    one contribution (``_centred_term_contribution``).  Either is recomputed
    from ``X`` if a row's fallback needs it.
    """
    spec = term["spec"]
    if not scores_centred(spec):
        return eta.add(lambda: _centred_term_contribution(term, X, beta_all, centre))
    beta = np.asarray(beta_all[term["beta_idx"]], dtype=np.float64).ravel()
    return eta.total.add_column(
        lambda: np.asarray(spec._centred_values(*_term_columns(term, X)), dtype=np.float64).ravel(),
        float(centre[0]),
        float(beta[0]),
    )


def _score_prediction_term_fast_discrete(
    model,
    term: dict[str, Any],
    X: EagerFrame,
    beta_all: NDArray,
) -> NDArray[np.floating]:
    """Score one canonical term via the fast discrete approximation when available."""
    fast_discrete = term["fast_discrete"]
    beta = beta_all[term["beta_idx"]]
    if fast_discrete is None:
        return _score_prediction_term_exact(term, X, beta_all)
    if fast_discrete["kind"] == "feature":
        return _score_feature_fast_discrete(term, X, beta)
    return _score_interaction_fast_discrete(model, term, X, beta)


def _predict_eta(
    model,
    X,
    offset: NDArray | None,
    *,
    fast_discrete: bool,
    random_effects: str,
    stabilize: bool = True,
    fitted: bool = False,
    warn: bool = True,
) -> NDArray[np.floating]:
    """Predict the raw or stabilized linear predictor on canonical blocks.

    ``fitted`` scores every term at the fit's own coefficients, the fit's
    linear predictor on its training rows, for the library's evaluations of
    the fit (screening's working score, random-effect reporting, the
    discretization deltas); ``predict`` instead treats ``sz`` levels the data
    identify only in part (#432, ``FactorSmooth._identified_blocks``), and
    warns of them unless ``warn`` is false.
    """
    if random_effects not in ("conditional", "population"):
        raise ValueError(
            f"random_effects must be 'conditional' or 'population', got {random_effects!r}"
        )

    from superglm.features.factor_smooth import SZ_POPULATION_PREDICTION, FactorSmooth
    from superglm.features.random_effect import RandomEffect

    frame = as_eager_frame(X)
    plan = _prediction_plan(model)
    required_columns = tuple(
        dict.fromkeys(
            [term["name"] for term in plan["features"]]
            + [parent for term in plan["interactions"] for parent in term["parent_names"]]
        )
    )
    frame.require_columns(required_columns)
    validate_x_columns(frame, required_columns)
    offset = validate_prediction_offset(offset, len(frame))
    beta_all = model.result.beta
    # The fit's centred predictor (one-engine design §3.8) when it carries one:
    # a dense column's offset would otherwise cancel between X beta and the
    # raw intercept and return the rounding the fit avoided.
    intercept, centre, intercept_lo = prediction_centred_state(model.result)
    eta = EtaSum(
        len(frame),
        intercept,
        intercept_lo,
        compensated=centre is not None
        and bool(getattr(model.result, "centred_sum_compensated", False)),
    )

    scorer = _score_prediction_term_fast_discrete if fast_discrete else _score_prediction_term_exact

    def score(term: dict[str, Any]) -> None:
        if centre is not None:
            block = centre[term["beta_idx"]]
            if np.any(block != 0.0):
                if eta.compensated:
                    _add_centred_term(eta, term, frame, beta_all, block)
                else:
                    eta.add(_centred_term_contribution(term, frame, beta_all, block))
                return
        # A callable, so a compensated sum need not hold the term's rows.
        if fast_discrete:
            eta.add(lambda: scorer(model, term, frame, beta_all))
        else:
            eta.add(lambda: scorer(term, frame, beta_all))

    for term in plan["features"]:
        if random_effects == "population" and isinstance(term["spec"], RandomEffect):
            term["spec"].validate_prediction_values(frame.column_array(term["name"]))
            continue
        score(term)

    if not fitted:
        from superglm.model.fit_ops import _ensure_factor_smooth_levels_recorded

        _ensure_factor_smooth_levels_recorded(model)
    unidentified: list[str] = []
    for term in plan["interactions"]:
        spec = term["spec"]
        if not fitted and isinstance(spec, FactorSmooth) and spec._has_population_offset:
            contribution, named = _score_unidentified_factor_smooth(
                term, frame, beta_all, population=random_effects == "population"
            )
            eta += contribution
            if named:
                unidentified.append(f"term {term['name']!r} levels {', '.join(map(str, named))}")
            continue
        if random_effects == "population" and isinstance(spec, FactorSmooth):
            left_name, right_name = term["parent_names"]
            spec.validate_population_prediction_values(
                frame.column_array(left_name),
                frame.column_array(right_name),
            )
            continue
        score(term)
    if unidentified and warn:
        warnings.warn(
            SZ_POPULATION_PREDICTION + "; ".join(unidentified) + ".",
            UserWarning,
            stacklevel=_PREDICTION_WARNING_STACKLEVEL,
        )

    total = eta.finish(consume=True)
    if offset is not None:
        total = total + offset
    return stabilize_eta(total, model._link) if stabilize else total


def predict_eta_exact(
    model,
    X: EagerFrame | FrameLike,
    offset: NDArray | None = None,
    *,
    random_effects: str = "conditional",
    fitted: bool = False,
    warn: bool = True,
) -> NDArray[np.floating]:
    """Predict the stabilized linear predictor through the exact canonical contract.

    ``fitted`` and ``warn`` as ``_predict_eta``'s.
    """
    return _predict_eta(
        model,
        X,
        offset,
        fast_discrete=False,
        random_effects=random_effects,
        fitted=fitted,
        warn=warn,
    )


def predict_eta_raw_exact(
    model,
    X: EagerFrame | FrameLike,
    offset: NDArray | None = None,
    *,
    random_effects: str = "conditional",
) -> NDArray[np.floating]:
    """Predict an unstabilized linear predictor through the exact design path."""
    return _predict_eta(
        model,
        X,
        offset,
        fast_discrete=False,
        random_effects=random_effects,
        stabilize=False,
    )


def predict_eta_fast_discrete(
    model,
    X: FrameLike,
    offset: NDArray | None = None,
    *,
    random_effects: str = "conditional",
) -> NDArray[np.floating]:
    """Predict the stabilized linear predictor through the fast discrete contract."""
    return _predict_eta(
        model,
        X,
        offset,
        fast_discrete=True,
        random_effects=random_effects,
    )


def _eta_to_mu(model, eta: NDArray[np.floating]) -> NDArray:
    """Map stabilized eta to the public response scale."""
    return clip_mu(model._link.inverse(eta), model._distribution)


def predict_exact(
    model,
    X: FrameLike,
    offset: NDArray | None = None,
    *,
    random_effects: str = "conditional",
    warn: bool = True,
) -> NDArray:
    """Predict the response mean through the exact canonical contract."""
    return _eta_to_mu(
        model,
        predict_eta_exact(model, X, offset, random_effects=random_effects, warn=warn),
    )


def predict_fast_discrete(
    model,
    X: FrameLike,
    offset: NDArray | None = None,
    *,
    random_effects: str = "conditional",
) -> NDArray:
    """Predict the response mean through the fast discrete contract."""
    return _eta_to_mu(
        model,
        predict_eta_fast_discrete(model, X, offset, random_effects=random_effects),
    )


def resolve_penalty(
    penalty: Penalty | str | None,
    lambda1: SelectionPenalty,
    penalty_features: str | list[str] | None = None,
) -> Penalty:
    """Convert string shorthand / None to a Penalty object."""
    resolved_lambda1 = normalize_selection_penalty(lambda1)
    if penalty is None:
        return cast(Penalty, GroupLasso(lambda1=resolved_lambda1, features=penalty_features))
    if isinstance(penalty, str):
        if penalty not in _PENALTY_SHORTCUTS:
            raise ValueError(
                f"Unknown penalty '{penalty}'. "
                f"Use one of {list(_PENALTY_SHORTCUTS)} or pass a Penalty object."
            )
        return cast(
            Penalty,
            _PENALTY_SHORTCUTS[penalty](
                lambda1=resolved_lambda1,
                features=penalty_features,
            ),
        )
    if lambda1 is not None:
        raise ValueError(
            "Cannot set 'selection_penalty' when passing a Penalty object directly. "
            "Set lambda1 on the Penalty object instead."
        )
    if penalty_features is not None:
        raise ValueError(
            "Cannot set 'penalty_features' when passing a Penalty object directly. "
            "Set features on the Penalty object instead."
        )
    owned_penalty = copy.deepcopy(penalty)
    cast(Any, owned_penalty).lambda1 = normalize_selection_penalty(owned_penalty.lambda1)
    return owned_penalty


def normalize_selection_penalty(value: object) -> SelectionPenalty:
    """Normalize explicit selection intent without choosing a fitted value."""
    if value is None:
        return None
    if isinstance(value, str):
        if value == "auto":
            return value
        raise ValueError(_SELECTION_PENALTY_ERROR)
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(_SELECTION_PENALTY_ERROR)
    try:
        numeric = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(_SELECTION_PENALTY_ERROR) from exc
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(_SELECTION_PENALTY_ERROR)
    return numeric


def resolve_knots(model, spline_cols: list[str]) -> dict[str, int]:
    """Map spline column names to their n_knots values."""
    if not spline_cols:
        return {}
    if isinstance(model._n_knots, int):
        return {col: model._n_knots for col in spline_cols}
    if len(model._n_knots) != len(spline_cols):
        raise ValueError(
            f"n_knots has length {len(model._n_knots)} but splines "
            f"has length {len(spline_cols)}. Must match or pass a single int."
        )
    return dict(zip(spline_cols, model._n_knots))


def validate_factor_smooth_configuration(model, *, features_resolved: bool) -> None:
    """Validate factor-smooth geometry after explicit or inferred features exist."""
    from superglm.features.categorical import Categorical
    from superglm.features.factor_smooth import FactorSmooth
    from superglm.features.random_effect import RandomEffect
    from superglm.features.spline import _SplineBase

    terms = [
        spec
        for name in model._interaction_order
        if isinstance((spec := model._interaction_specs[name]), FactorSmooth)
    ]
    seen: dict[tuple[str, str], str] = {}
    for term in terms:
        pair = (term.variable, term.group)
        if pair in seen:
            raise ValueError(
                f"FactorSmooth pair {pair!r} is configured more than once "
                f"({seen[pair]!r} and {term.name!r})."
            )
        seen[pair] = term.name

    if not features_resolved:
        return

    for term in terms:
        group_spec = model._specs.get(term.group)
        if isinstance(group_spec, (Categorical, RandomEffect)):
            raise ValueError(
                f"FactorSmooth {term.name!r} group {term.group!r} duplicates "
                "the constant null-space group-intercept geometry of "
                f"{type(group_spec).__name__} on the same column; remove that "
                "group main effect and use an explicit features map containing "
                "only the intended main effects."
            )
        if term.basis == "sz" and not isinstance(
            model._specs.get(term.variable),
            _SplineBase,
        ):
            raise ValueError(
                f"FactorSmooth {term.name!r} with basis='sz' requires a global "
                f"Spline for {term.variable!r}; use "
                f"features={{{term.variable!r}: Spline(...)}}."
            )


def init_model(
    model,
    family: str | Distribution = "poisson",
    link: str | Link | None = None,
    penalty: Penalty | str | None = None,
    lambda1: SelectionPenalty = None,
    lambda2: float = 0.1,
    penalty_features: str | list[str] | None = None,
    features: Mapping[Hashable, FeatureSpec] | None = None,
    splines: list[str] | None = None,
    n_knots: int | list[int] = 10,
    degree: int = 3,
    categorical_base: str = "most_exposed",
    interactions: list[tuple[str, str] | object] | None = None,
    active_set: bool = False,
    direct_solve: str = "auto",
    discrete: bool = False,
    n_bins: int | dict[str, int] = 256,
    tol: float = 1e-6,
    max_iter: int = 100,
    convergence: str = "deviance",
    retain_fit_state: bool = True,
    separation: str = "warn",
    group_pricing: str = "rank",
    weight_semantics: str = "prior",
    n_jobs: int | str = "auto",
    max_memory: int | str = "auto",
):
    """Initialize model state (body of SuperGLM.__init__)."""
    if features is not None and splines is not None:
        raise ValueError(
            "Cannot set both 'features' and 'splines'. "
            "Use 'features' for explicit specs or 'splines' for auto-detect."
        )
    # Constructor inputs become model-owned immediately.  Learned fit state is
    # built from a second private template, never by mutating caller objects.
    model.family = family
    model.link = link
    model.penalty = resolve_penalty(penalty, lambda1, penalty_features)
    model.lambda2 = lambda2
    model._features_explicit = features is not None
    model._splines = None if splines is None else list(splines)
    model._n_knots = copy.deepcopy(n_knots)
    model._degree = degree
    model._categorical_base = categorical_base
    model._active_set = active_set
    if direct_solve not in ("auto", "gram", "qr", "structured"):
        raise ValueError(
            f"direct_solve must be 'auto', 'gram', 'qr', or 'structured', got {direct_solve!r}"
        )
    model._direct_solve = direct_solve
    model._discrete = discrete
    model._n_bins = copy.deepcopy(n_bins)
    if group_pricing not in ("rank", "spanned"):
        raise ValueError(f"group_pricing must be 'rank' or 'spanned', got {group_pricing!r}")
    model._group_pricing = group_pricing
    model._weight_semantics = validate_weight_semantics(weight_semantics)
    from superglm._parallel import validate_max_memory, validate_n_jobs

    model._n_jobs = validate_n_jobs(n_jobs)
    model._max_memory = validate_max_memory(max_memory)
    model._tol = tol
    model._max_iter = max_iter
    model._retain_fit_state = bool(retain_fit_state)
    if convergence not in ("deviance", "coefficients"):
        raise ValueError(f"convergence must be 'deviance' or 'coefficients', got {convergence!r}")
    if convergence == "coefficients":
        import warnings

        warnings.warn(
            "convergence='coefficients' is experimental. Near-separated levels "
            "have no finite MLE, so coefficient-based convergence may not "
            "terminate or may produce numerically unstable results. "
            "Use convergence='deviance' (default) for production fits.",
            UserWarning,
            stacklevel=3,
        )
    model._convergence = convergence
    from superglm.diagnostics.separation import validate_separation_mode

    model._separation = validate_separation_mode(separation)

    model._specs: dict[Hashable, FeatureSpec] = {}
    model._feature_order: list[Hashable] = []
    model._groups: list[GroupSlice] = []
    model._distribution: Distribution | None = None
    model._link: Link | None = None
    model._result: PIRLSResult | None = None
    model._solver_result: PIRLSResult | None = None
    model._linear_system_state = None
    model._reporting_support_state = None
    model._dm: DesignMatrix | None = None
    model._fit_weights: NDArray | None = None
    model._fit_offset: NDArray | None = None
    model._fit_used_offset = False
    model._fit_used_weights = False
    model._fit_stats: FitStats | None = None
    model._runtime_canonical_state: dict[str, Any] | None = None
    model._nb_profile_result = None
    model._tweedie_profile_result = None
    model._last_fit_meta: dict[str, Any] | None = None
    model._monotone_repairs: dict = {}
    model._prediction_plan = None
    model._fast_prediction_state = None
    model._fit_mu: NDArray | None = None
    model._fit_null_mu: NDArray | None = None
    model._fit_X_ref = None
    model._fit_y_ref = None
    model._fit_sample_weight_ref = None
    model._fit_offset_ref = None
    model._fit_data_guard = None
    model._fit_geometry_guard = None
    model._fit_metrics_cache = None
    model._fit_metrics_cache_signature = None
    model._summary_cache = None

    # Interaction support. Tuple interactions are resolved after feature
    # construction; explicit specs already own their parent-column contract.
    model._interaction_specs: dict[str, Any] = {}
    model._interaction_order: list[str] = []
    pending_interactions: list[tuple[str, str]] = []
    explicit_interactions: list[Any] = []
    for interaction in interactions or ():
        if (
            isinstance(interaction, tuple)
            and len(interaction) == 2
            and all(isinstance(name, str) for name in interaction)
        ):
            pending_interactions.append(interaction)
            continue
        parent_names = getattr(interaction, "parent_names", None)
        interaction_name = getattr(interaction, "name", None)
        if (
            not isinstance(parent_names, tuple)
            or len(parent_names) != 2
            or not all(isinstance(parent, str) and parent for parent in parent_names)
            or not isinstance(interaction_name, str)
            or not interaction_name
        ):
            raise TypeError(
                "interactions entries must be (left, right) tuples or explicit "
                "interaction specs with parent_names and name"
            )
        if interaction_name in model._interaction_specs or any(
            existing.name == interaction_name for existing in explicit_interactions
        ):
            raise ValueError(f"Interaction already added: {interaction_name}")
        explicit_interactions.append(interaction)
    model._pending_interactions = tuple(pending_interactions)

    # Register explicit features dict
    if features is not None:
        for name, spec in features.items():
            model._specs[name] = copy.deepcopy(spec)
            model._feature_order.append(name)

    for interaction in explicit_interactions:
        owned = copy.deepcopy(interaction)
        model._interaction_specs[owned.name] = owned
        model._interaction_order.append(owned.name)

    validate_term_name_namespace(model._specs, model._interaction_specs)
    validate_factor_smooth_configuration(
        model,
        features_resolved=model._splines is None,
    )

    from superglm.model.fit_state import ModelConfig

    model._config_revision = 0
    model._fit_revision = 0
    model._fit_state = None
    model._selection_penalty_fitted = None
    model._distribution_fitted = None
    model._resolved_penalty = None
    model._config = ModelConfig.capture(model)


def clone_without_features(
    model,
    drop: set[str],
    *,
    lambda1: float | None = ...,  # sentinel: ... means "keep current"
    lambda2: float | dict[str, float] | None = ...,
):
    """Create a new SuperGLM with a subset of features removed.

    Copies family, link, penalty type, and solver options. Interactions
    whose parents include a dropped feature are also removed.
    """
    keep_features = {n: s for n, s in model._specs.items() if n not in drop}

    # Filter interactions: drop any whose parent is being dropped
    from superglm.features.factor_smooth import FactorSmooth

    keep_interactions: list[tuple[str, str] | object] = []
    # Check resolved interactions (fitted model)
    for iname in model._interaction_order:
        ispec = model._interaction_specs[iname]
        p1, p2 = ispec.parent_names
        if p1 not in drop and p2 not in drop:
            keep_interactions.append(
                copy.deepcopy(ispec) if isinstance(ispec, FactorSmooth) else (p1, p2)
            )
    # Check pending interactions (unfitted model)
    for p1, p2 in model._pending_interactions:
        if p1 not in drop and p2 not in drop:
            keep_interactions.append((p1, p2))

    # Resolve lambda1
    source_penalty = fitted_penalty(model)
    if lambda1 is ...:
        lam1 = source_penalty.lambda1
    else:
        lam1 = lambda1

    new_penalty = copy.deepcopy(source_penalty)
    new_penalty.lambda1 = lam1

    # Deep-copy specs so the new model doesn't share mutable state
    new_features = {n: copy.deepcopy(s) for n, s in keep_features.items()}

    fit_state = getattr(model, "_fit_state", None)
    if fit_state is None:
        source_family = configured_family(model)
        source_link = configured_link(model)
    else:
        source_family = fit_state.distribution
        source_link = fit_state.projections.get("_link", model._link)

    new_model = type(model)(
        family=source_family,
        link=source_link,
        penalty=new_penalty,
        features=new_features,
        interactions=keep_interactions if keep_interactions else None,
        active_set=model._active_set,
        direct_solve=model._direct_solve,
        discrete=model._discrete,
        n_bins=model._n_bins,
        tol=model._tol,
        max_iter=model._max_iter,
        convergence=model._convergence,
        retain_fit_state=model._retain_fit_state,
        # A clone is the SAME model with fewer terms, so it must be fitted
        # under the same likelihood. Falling through to the constructor
        # default silently turned every frequency-contract model into a prior
        # one the moment a term was dropped -- which is the path `drop1`,
        # term importance and `refit_unpenalised` all take.
        weight_semantics=model_weight_semantics(model),
        # Its other model-level rules too: falling through, `separation="error"`
        # became "warn" and `group_pricing="spanned"` became "rank" on every
        # clone (and the editor's Refit and `Structure.apply`, which clone
        # here), refitting a different pricing rule. A model pickled before
        # `group_pricing` existed keeps "spanned", as `model_build_design_matrix`
        # reads it.
        separation=getattr(model, "_separation", "warn"),
        group_pricing=getattr(model, "_group_pricing", "spanned"),
    )
    # The universes `bind_levels` bound, for the features kept.
    bindings = {
        name: binding
        for name, binding in dict(getattr(model, "_level_bindings", None) or ()).items()
        if name not in drop
    }
    if bindings:
        stored = tuple(bindings.items())
        new_model._level_bindings = stored
        new_model._config = new_model._config.with_value(level_bindings=stored)

    # Resolve lambda2
    if lambda2 is ...:
        source_lambda2 = fitted_lambda2(model)
        if isinstance(source_lambda2, dict):
            # Filter REML lambdas to remaining groups
            new_model.lambda2 = {
                k: v
                for k, v in source_lambda2.items()
                if not any(k == d or k.startswith(f"{d}:") for d in drop)
            }
        else:
            new_model.lambda2 = source_lambda2
    elif lambda2 is None:
        new_model.lambda2 = 0.0
    else:
        new_model.lambda2 = lambda2

    return new_model


def auto_detect(model, X: EagerFrame, sample_weight: NDArray | None) -> None:
    """Auto-detect feature types from native dataframe columns."""
    spline_cols = model._splines or []
    knots_map = resolve_knots(model, spline_cols)
    auto_detect_features(
        X,
        sample_weight,
        spline_cols=spline_cols,
        knots_map=knots_map,
        degree=model._degree,
        categorical_base=model._categorical_base,
        specs=model._specs,
        feature_order=model._feature_order,
    )
    validate_term_name_namespace(model._specs, model._interaction_specs)
    validate_factor_smooth_configuration(model, features_resolved=True)


def model_add_interaction(model, feat1: str, feat2: str, name: str | None = None, **kwargs) -> None:
    """Register an interaction between two already-registered features."""
    add_interaction(
        feat1,
        feat2,
        specs=model._specs,
        interaction_specs=model._interaction_specs,
        interaction_order=model._interaction_order,
        name=name,
        **kwargs,
    )
    if hasattr(model, "_config"):
        model._config = model._config.with_value(
            interactions=tuple(model._pending_interactions),
            interaction_templates=tuple(
                (interaction_name, copy.deepcopy(model._interaction_specs[interaction_name]))
                for interaction_name in model._interaction_order
            ),
            interaction_order=tuple(model._interaction_order),
        )
        model._config_revision += 1


def selection_penalty_is_active(model, *, force: bool = False) -> bool:
    """Whether a nonzero lambda1 will reach this fit's groups.

    Read at *build* time, so it cannot consult the resolved value --
    ``"auto"`` has not been calibrated yet, and under ``fit_path`` the grid
    is not known until the design exists.  Only the yes/no answer is needed,
    and it is the same for every point of a path, so callers that sweep
    lambda1 pass ``force=True`` rather than the value they start from.
    """
    if force:
        return True
    intent = normalize_selection_penalty(getattr(configured_penalty(model), "lambda1", None))
    return intent is not None and intent != 0.0


def model_build_design_matrix(
    model,
    X: EagerFrame | FrameLike,
    y: NDArray,
    sample_weight: NDArray,
    offset: NDArray | None,
    *,
    selection_active: bool | None = None,
) -> tuple[NDArray, NDArray, NDArray | None]:
    """Build features, groups, design matrix.

    Sets model._dm, model._groups, model._distribution, model._link.
    Returns (y, sample_weight, offset) as float64 arrays.

    ``selection_active`` says whether a nonzero lambda1 will reach this fit.
    It suppresses the aliased-cell half of categorical interaction pruning,
    which is only free while the solver's rank convention chooses the
    representative of a cross-group rank deficiency; a group penalty chooses
    a different one.  ``None`` reads the model's configured intent, which is
    right for every caller that does not sweep lambda1 itself.
    """
    if selection_active is None:
        selection_active = selection_penalty_is_active(model)
    frame = as_eager_frame(X)
    pending_interactions = list(model._pending_interactions)
    result = build_design_matrix(
        frame,
        y,
        sample_weight,
        offset,
        family=configured_family(model),
        link_spec=configured_link(model),
        specs=model._specs,
        feature_order=model._feature_order,
        interaction_specs=model._interaction_specs,
        interaction_order=model._interaction_order,
        pending_interactions=pending_interactions,
        model_discrete=model._discrete,
        n_bins_config=model._n_bins,
        lambda2=configured_lambda2(model),
        level_bindings=(
            dict(model._level_bindings) if getattr(model, "_level_bindings", None) else None
        ),
        alias_prune=not selection_active,
        separation=getattr(model, "_separation", "warn"),
        selection_penalty=configured_penalty(model) if selection_active else None,
        # Models pickled before ``group_pricing`` existed keep the behaviour
        # they were fitted under, matching the ModelConfig unpickle backfill.
        group_pricing=getattr(model, "_group_pricing", "spanned"),
        weight_semantics=model_weight_semantics(model),
    )
    model._distribution = result.distribution
    model._link = result.link
    # The builder compiles into an owned clone rather than mutating the caller's
    # spec graph in place, so learned state is adopted by rebinding here.
    model._specs = dict(result.compiled.specs)
    model._feature_order = list(result.compiled.feature_order)
    model._interaction_specs = dict(result.compiled.interaction_specs)
    model._interaction_order = list(result.compiled.interaction_order)
    model._pending_interactions = ()
    model._groups = list(result.compiled.groups)
    _validate_group_feature_names(model, result.groups)
    validate_penalty_features(configured_penalty(model), result.groups)
    model._dm = result.dm
    # Every fit path funnels through here, so this is the first point the
    # design width is known: wide designs release the small-p BLAS cap.
    from superglm._blas_threads import allow_wide_design

    allow_wide_design(result.dm.p)
    return result.y, result.sample_weight, result.offset


def compute_lambda_max(model, y, weights):
    """Smallest lambda1 at which all groups are zeroed (null model).

    Must agree with the solver's own zeroing rule.  The block-coordinate step
    zeroes group ``g`` when ``||grad_g|| <= lambda1 * g.weight``
    (``pirls.py`` radial threshold, ``GroupLasso.prox_group``), with
    ``grad_g = -X_g' W r``.  Expanding the IRLS working quantities,
    ``W r = sample_weight * (dmu/deta) * (y - mu) / V(mu)``, so the null-model
    score carries a family factor that is 1 only for canonical links.  The
    solver's objective is unnormalised, so no row-count division belongs here.
    """
    from superglm.distributions import initial_mean
    from superglm.screening import working_score
    from superglm.solvers.mode_score import (
        dense_centred_rmatvec,
        dense_columns,
        prior_weighted_centre,
    )

    mu_null = np.atleast_1d(np.asarray(initial_mean(y, weights, model._distribution), float))
    eta_null = model._link.link(mu_null)
    score = working_score(y, mu_null, eta_null, weights, model._distribution, model._link)
    grad = model._dm.rmatvec(score)
    # A dense column's score is read about its prior-weighted centre, by type,
    # as the proximal solver reads it about its pair (issue #430; the null W is
    # the prior times a constant, so the two centres agree).  sum(score) is zero
    # only to rounding, and raw, a column at 1e12 multiplied that rounding by
    # its offset (lambda_max read 267.0078125 for 267).
    dense = dense_columns(model._dm)
    if np.any(dense):
        prior = np.ones(model._dm.n) if weights is None else np.asarray(weights, dtype=float)
        centre = prior_weighted_centre(model._dm, prior)
        grad = np.where(dense, dense_centred_rmatvec(model._dm, score, centre), grad)
    penalty = configured_penalty(model)
    # GroupLasso thresholds at lambda1 * w_g; the elastic-net family scales that
    # by its L1 share alpha (pirls.py radial threshold), so the lambda that
    # zeroes a group is correspondingly larger.
    from superglm.penalties.sparse_group_lasso import SparseGroupLasso

    is_sparse_group = isinstance(penalty, SparseGroupLasso)
    alpha = 1.0 if type(penalty) is GroupLasso else float(getattr(penalty, "alpha", 1.0))
    if alpha <= 0.0 and not is_sparse_group:
        # e.g. elastic-net alpha=0 is ridge: nothing ever zeroes.
        return 0.0
    lmax = 0.0
    for g in model._groups:
        if not penalty_targets_group(penalty, g):
            continue
        grad_g = grad[g.sl]
        if is_sparse_group:
            lmax = max(lmax, _sparse_group_zero_lambda(grad_g, float(g.weight), alpha))
        else:
            lmax = max(lmax, np.linalg.norm(grad_g) / (g.weight * alpha))
    return lmax


def _sparse_group_zero_lambda(grad_g: NDArray, weight: float, alpha: float) -> float:
    """Smallest lambda that zeroes one group under the sparse-group KKT rule.

    The composite zero condition is ``||soft(grad_g, lambda*alpha)||_2 <=
    lambda*(1-alpha)*weight`` (``pirls.py`` prox: L1 soft-threshold first,
    then the radial group threshold) — not the elastic-net division.  At
    alpha=0 this reduces to the pure group-lasso ``||grad_g||/weight``; at
    alpha=1 to the pure-L1 ``max|grad_g|``.  In between the left side is
    nonincreasing and the right side increasing in lambda, so the boundary
    is the unique root, found by bisection to float accuracy.
    """
    magnitudes = np.abs(np.asarray(grad_g, dtype=np.float64))
    if magnitudes.size == 0:
        return 0.0
    if alpha <= 0.0:
        return float(np.linalg.norm(magnitudes)) / weight
    if alpha >= 1.0:
        return float(magnitudes.max())
    lo, hi = 0.0, float(magnitudes.max()) / alpha  # soft() == 0 at hi: zeroed
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        survived = np.maximum(magnitudes - mid * alpha, 0.0)
        if np.linalg.norm(survived) > mid * (1.0 - alpha) * weight:
            lo = mid
        else:
            hi = mid
    return hi


def resolve_selection_penalty_for_fit(model, penalty: Penalty, y, weights) -> float:
    """Resolve one ordinary-fit selection setting on attempt-owned state."""
    intent = normalize_selection_penalty(penalty.lambda1)
    if intent == "auto":
        resolved = float(compute_lambda_max(model, y, weights) * 0.1)
    elif intent is None:
        resolved = 0.0
    else:
        resolved = float(intent)
    cast(Any, penalty).lambda1 = resolved
    return resolved


def validate_selection_penalty_for_reml(penalty: Penalty) -> None:
    """Reject selection intent before any REML or profile work starts."""
    intent = normalize_selection_penalty(penalty.lambda1)
    if intent == "auto" or (intent is not None and intent > 0.0):
        raise ValueError(
            "fit_reml() does not support selection penalties; use None or 0.0, "
            "or use fit()/fit_path() for sparse selection."
        )


def resolve_selection_penalty_for_reml(penalty: Penalty) -> float:
    """Resolve REML's validated no-selection setting to numeric zero."""
    validate_selection_penalty_for_reml(penalty)
    cast(Any, penalty).lambda1 = 0.0
    return 0.0


def model_has_lambda1_targets(model) -> bool:
    """Whether the lambda1 penalty applies to any fitted group."""
    return penalty_has_targets(configured_penalty(model), model._groups)


def rebuild_dm_with_lambdas(
    model, lambdas: dict[str, float], sample_weight: NDArray
) -> DesignMatrix:
    """Rebuild design matrix with per-group smoothing lambdas."""
    return rebuild_design_matrix_with_lambdas(
        model._dm,
        model._groups,
        lambdas,
        sample_weight,
        configured_lambda2(model),
    )


def predict(
    model,
    X: FrameLike,
    offset: NDArray | None = None,
    *,
    random_effects: str = "conditional",
) -> NDArray:
    """Predict the response mean for new data."""
    return predict_exact(model, X, offset, random_effects=random_effects)
