"""Constructor, prediction, and core helpers for SuperGLM."""

from __future__ import annotations

import copy
import logging
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
)
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


def _group_beta_indices(groups: list[GroupSlice], feature_name: str) -> NDArray[np.intp]:
    """Concatenate coefficient indices for one feature or interaction."""
    idx = [
        np.arange(g.start, g.end, dtype=np.intp) for g in groups if g.feature_name == feature_name
    ]
    if not idx:
        raise KeyError(f"No fitted groups found for feature {feature_name!r}")
    if len(idx) == 1:
        return idx[0]
    return np.concatenate(idx)


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


def _score_prediction_term_exact(
    term: dict[str, Any],
    X: EagerFrame,
    beta_all: NDArray,
) -> NDArray[np.floating]:
    """Score one canonical term exactly on the requested rows."""
    beta = beta_all[term["beta_idx"]]
    if term["kind"] == "feature":
        return _score_feature(term["spec"], X.column_array(term["name"]), beta)

    left_name, right_name = term["parent_names"]
    return _score_interaction(
        term["spec"],
        X.column_array(left_name),
        X.column_array(right_name),
        beta,
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
) -> NDArray[np.floating]:
    """Predict the stabilized linear predictor on exact or fast-discrete blocks."""
    if random_effects not in ("conditional", "population"):
        raise ValueError(
            f"random_effects must be 'conditional' or 'population', got {random_effects!r}"
        )

    from superglm.features.factor_smooth import FactorSmooth
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
    beta_all = model.result.beta
    eta = np.full(len(frame), model.result.intercept, dtype=np.float64)

    scorer = _score_prediction_term_fast_discrete if fast_discrete else _score_prediction_term_exact

    for term in plan["features"]:
        if random_effects == "population" and isinstance(term["spec"], RandomEffect):
            term["spec"].validate_prediction_values(frame.column_array(term["name"]))
            continue
        if fast_discrete:
            eta += scorer(model, term, frame, beta_all)
        else:
            eta += scorer(term, frame, beta_all)

    for term in plan["interactions"]:
        if random_effects == "population" and isinstance(term["spec"], FactorSmooth):
            left_name, right_name = term["parent_names"]
            term["spec"].validate_prediction_values(
                frame.column_array(left_name),
                frame.column_array(right_name),
            )
            continue
        if fast_discrete:
            eta += scorer(model, term, frame, beta_all)
        else:
            eta += scorer(term, frame, beta_all)

    if offset is not None:
        eta = eta + np.asarray(offset, dtype=np.float64)
    return stabilize_eta(eta, model._link)


def predict_eta_exact(
    model,
    X: EagerFrame | FrameLike,
    offset: NDArray | None = None,
    *,
    random_effects: str = "conditional",
) -> NDArray[np.floating]:
    """Predict the stabilized linear predictor through the exact canonical contract."""
    return _predict_eta(
        model,
        X,
        offset,
        fast_discrete=False,
        random_effects=random_effects,
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
) -> NDArray:
    """Predict the response mean through the exact canonical contract."""
    return _eta_to_mu(
        model,
        predict_eta_exact(model, X, offset, random_effects=random_effects),
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


def init_model(
    model,
    family: str | Distribution = "poisson",
    link: str | Link | None = None,
    penalty: Penalty | str | None = None,
    lambda1: SelectionPenalty = None,
    lambda2: float = 0.1,
    penalty_features: str | list[str] | None = None,
    features: dict[str, FeatureSpec] | None = None,
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

    model._specs: dict[str, FeatureSpec] = {}
    model._feature_order: list[str] = []
    model._groups: list[GroupSlice] = []
    model._distribution: Distribution | None = None
    model._link: Link | None = None
    model._result: PIRLSResult | None = None
    model._solver_result: PIRLSResult | None = None
    model._linear_system_state = None
    model._dm: DesignMatrix | None = None
    model._fit_weights: NDArray | None = None
    model._fit_offset: NDArray | None = None
    model._fit_used_offset = False
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

    from superglm.features.factor_smooth import FactorSmooth
    from superglm.features.random_effect import RandomEffect

    for interaction in explicit_interactions:
        owned = copy.deepcopy(interaction)
        if isinstance(owned, FactorSmooth) and isinstance(
            model._specs.get(owned.group),
            RandomEffect,
        ):
            raise ValueError(
                f"FactorSmooth group {owned.group!r} duplicates the constant null-space "
                "geometry of the RandomEffect on the same column."
            )
        model._interaction_specs[owned.name] = owned
        model._interaction_order.append(owned.name)

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
    )

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


def model_build_design_matrix(
    model,
    X: EagerFrame | FrameLike,
    y: NDArray,
    sample_weight: NDArray,
    offset: NDArray | None,
) -> tuple[NDArray, NDArray, NDArray | None]:
    """Build features, groups, design matrix.

    Sets model._dm, model._groups, model._distribution, model._link.
    Returns (y, sample_weight, offset) as float64 arrays.
    """
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
    )
    model._distribution = result.distribution
    model._link = result.link
    model._pending_interactions = ()
    model._groups = result.groups
    validate_penalty_features(configured_penalty(model), result.groups)
    model._dm = result.dm
    return result.y, result.sample_weight, result.offset


def compute_lambda_max(model, y, weights):
    """Smallest lambda1 at which all groups are zeroed (null model)."""
    from superglm.distributions import initial_mean

    mu_null = initial_mean(y, weights, model._distribution)
    residual = weights * (y - mu_null)
    grad = model._dm.rmatvec(residual)
    n = model._dm.n
    lmax = 0.0
    for g in model._groups:
        if not penalty_targets_group(configured_penalty(model), g):
            continue
        lmax = max(lmax, np.linalg.norm(grad[g.sl]) / g.weight)
    return lmax / n


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
