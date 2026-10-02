"""Configuration templates and the atomic fitted-state installation boundary."""

from __future__ import annotations

import copy
import math
from collections.abc import Hashable, Iterator, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from superglm.solvers.dispersion import model_weight_semantics

if TYPE_CHECKING:
    from superglm.types import LevelBinding


class FrozenMapping(Mapping[str, object]):
    """Pickle-safe read-only mapping used by immutable fitted state."""

    __slots__ = ("__mapping",)

    def __init__(self, values: Mapping[str, object]) -> None:
        from types import MappingProxyType

        self.__mapping = MappingProxyType(dict(values))

    def __getitem__(self, key: str) -> object:
        return self.__mapping[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.__mapping)

    def __len__(self) -> int:
        return len(self.__mapping)

    def __reduce__(self):
        return (type(self), (dict(self.__mapping),))


def configured_family(model):
    """Return model-owned family intent for internal computation."""
    return model._family_config


def configured_link(model):
    """Return model-owned link intent for internal computation."""
    return model._link_config


def configured_penalty(model):
    """Return model-owned selection-penalty intent for a new attempt."""
    return model._penalty_config


def configured_lambda2(model):
    """Return model-owned smoothing configuration for a new attempt."""
    return model._lambda2_config


def fitted_lambda2(model):
    """Return smoothing parameters belonging to the current fitted result."""
    state = getattr(model, "_fit_state", None)
    if state is not None and state.resolved_lambda2 is not None:
        return state.resolved_lambda2
    reml_lambdas = getattr(model, "_reml_lambdas", None)
    if reml_lambdas is not None:
        return reml_lambdas
    return configured_lambda2(model)


def fitted_penalty(model):
    """Return the resolved penalty belonging to the installed fitted state."""
    state = getattr(model, "_fit_state", None)
    if state is not None and state.resolved_penalty is not None:
        return state.resolved_penalty
    resolved = getattr(model, "_resolved_penalty", None)
    return resolved if resolved is not None else configured_penalty(model)


@dataclass(frozen=True)
class ModelConfig:
    """Small, model-owned constructor intent used to start every fit attempt."""

    family: object
    link: object
    penalty: object
    feature_templates: tuple[tuple[Hashable, object], ...]
    features_explicit: bool
    splines: tuple[str, ...] | None
    n_knots: int | tuple[int, ...]
    degree: int
    categorical_base: str
    interactions: tuple[tuple[str, str], ...]
    interaction_templates: tuple[tuple[str, object], ...]
    interaction_order: tuple[str, ...]
    lambda2: float | dict[str, float]
    active_set: bool
    direct_solve: str
    discrete: bool
    n_bins: int | dict[str, int]
    tol: float
    max_iter: int
    convergence: str
    retain_fit_state: bool
    # Class-attribute default doubles as the migration value for pickles
    # captured before the field existed (frozen dataclasses fall back to the
    # class attribute when the instance never stored one).
    separation: str = "warn"
    # Fit-machinery intent, not constructor API: a caller that knows the whole
    # frame (cross_validate) resolves one level universe per categorical term
    # and every attempt materialized from this configuration binds to it.
    level_bindings: tuple[tuple[Hashable, LevelBinding], ...] | None = None
    # Dimension the selection penalty and the fallback df ledger price a
    # group at when its spec emits fewer columns than the term spans.  New
    # models default to "rank"; configurations pickled before the field
    # existed restore to "spanned", the behaviour they were fitted under.
    group_pricing: str = "rank"
    # What ``sample_weight`` states about a row.  New models default to the EDM
    # prior-weight reading; configurations pickled before the field existed are
    # restored to whichever contract their family carried at the time, because
    # reproducing a recorded fit means reproducing the likelihood it was fitted
    # under, not adopting the new default.
    weight_semantics: str = "prior"

    def __getattr__(self, name: str) -> object:
        """Supply fields absent from models pickled before config migrations."""
        if name == "features_explicit":
            # This exactly matches the pre-field reconstruction contract:
            # non-empty templates were passed as ``features={...}``, while an
            # empty template set was reconstructed with ``features=None``.
            return bool(object.__getattribute__(self, "feature_templates"))
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore older pickles and materialize newly introduced config fields."""
        for name, value in state.items():
            object.__setattr__(self, name, value)
        if "features_explicit" not in state:
            object.__setattr__(
                self,
                "features_explicit",
                bool(state.get("feature_templates", ())),
            )
        if "weight_semantics" not in state:
            from superglm.distributions import resolve_distribution
            from superglm.solvers.dispersion import family_default_weight_semantics

            family = state.get("family", "poisson")
            try:
                distribution = resolve_distribution(family)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                distribution = None
            object.__setattr__(
                self,
                "weight_semantics",
                family_default_weight_semantics(distribution),
            )
        if "group_pricing" not in state:
            # Pre-field pickles were fitted under spanned pricing; restoring
            # them must reproduce the behaviour they recorded, not adopt the
            # new default.  This backfill is the only migration mechanism:
            # the field's dataclass default is a class attribute, so a
            # missing instance value would silently resolve to "rank".
            object.__setattr__(self, "group_pricing", "spanned")

    @classmethod
    def capture(cls, model) -> ModelConfig:
        """Capture configuration without retaining caller-owned mutable objects."""
        n_knots = model._n_knots
        if not isinstance(n_knots, int):
            n_knots = tuple(n_knots)
        splines = None if model._splines is None else tuple(model._splines)
        return cls(
            family=copy.deepcopy(configured_family(model)),
            link=copy.deepcopy(configured_link(model)),
            penalty=copy.deepcopy(configured_penalty(model)),
            feature_templates=tuple(
                (name, copy.deepcopy(model._specs[name])) for name in model._feature_order
            ),
            features_explicit=bool(getattr(model, "_features_explicit", bool(model._specs))),
            splines=splines,
            n_knots=n_knots,
            degree=int(model._degree),
            categorical_base=str(model._categorical_base),
            interactions=tuple(tuple(pair) for pair in model._pending_interactions),
            interaction_templates=tuple(
                (name, copy.deepcopy(model._interaction_specs[name]))
                for name in model._interaction_order
            ),
            interaction_order=tuple(model._interaction_order),
            lambda2=copy.deepcopy(configured_lambda2(model)),
            active_set=bool(model._active_set),
            direct_solve=str(model._direct_solve),
            discrete=bool(model._discrete),
            n_bins=copy.deepcopy(model._n_bins),
            tol=float(model._tol),
            max_iter=int(model._max_iter),
            convergence=str(model._convergence),
            retain_fit_state=bool(model._retain_fit_state),
            separation=str(getattr(model, "_separation", "warn")),
            level_bindings=copy.deepcopy(getattr(model, "_level_bindings", None)),
            group_pricing=str(getattr(model, "_group_pricing", "spanned")),
            weight_semantics=model_weight_semantics(model),
        )

    def with_value(self, **changes: object) -> ModelConfig:
        """Return a configuration revision with owned replacement values."""
        return replace(self, **{name: copy.deepcopy(value) for name, value in changes.items()})

    def constructor_kwargs(self) -> dict[str, object]:
        """Return an owned, exhaustive ``SuperGLM`` constructor configuration.

        Selection strength and feature targeting live on the resolved penalty
        template.  Passing them again as separate constructor arguments would
        conflict with the penalty-object API, so those slots are explicitly
        represented by ``None`` while the copied penalty carries their values.
        """
        return {
            "family": copy.deepcopy(self.family),
            "link": copy.deepcopy(self.link),
            "penalty": copy.deepcopy(self.penalty),
            "selection_penalty": None,
            "spline_penalty": copy.deepcopy(self.lambda2),
            "penalty_features": None,
            "features": (
                {name: copy.deepcopy(spec) for name, spec in self.feature_templates}
                if self.features_explicit
                else None
            ),
            "splines": (
                None if self.feature_templates or self.splines is None else list(self.splines)
            ),
            "n_knots": self.n_knots if isinstance(self.n_knots, int) else list(self.n_knots),
            "degree": self.degree,
            "categorical_base": self.categorical_base,
            "interactions": list(self.interactions) if self.interactions else None,
            "active_set": self.active_set,
            "direct_solve": self.direct_solve,
            "discrete": self.discrete,
            "n_bins": copy.deepcopy(self.n_bins),
            "tol": self.tol,
            "max_iter": self.max_iter,
            "convergence": self.convergence,
            "retain_fit_state": self.retain_fit_state,
            "separation": self.separation,
            "group_pricing": self.group_pricing,
            "weight_semantics": self.weight_semantics,
        }

    def materialize(self, model_type):
        """Create a fresh unfitted model from intent, never from prior fit state."""
        work_model = model_type.__new__(model_type)
        work_model.__dict__ = {
            "_family_config": copy.deepcopy(self.family),
            "_link_config": copy.deepcopy(self.link),
            "_penalty_config": copy.deepcopy(self.penalty),
            "_lambda2_config": copy.deepcopy(self.lambda2),
            "_features_explicit": self.features_explicit,
            "_splines": None if self.splines is None else list(self.splines),
            "_n_knots": self.n_knots if isinstance(self.n_knots, int) else list(self.n_knots),
            "_degree": self.degree,
            "_categorical_base": self.categorical_base,
            "_active_set": self.active_set,
            "_direct_solve": self.direct_solve,
            "_discrete": self.discrete,
            "_n_bins": copy.deepcopy(self.n_bins),
            "_group_pricing": self.group_pricing,
            "_weight_semantics": self.weight_semantics,
            "_tol": self.tol,
            "_max_iter": self.max_iter,
            "_retain_fit_state": self.retain_fit_state,
            "_convergence": self.convergence,
            "_separation": self.separation,
            "_specs": {name: copy.deepcopy(spec) for name, spec in self.feature_templates},
            "_level_bindings": copy.deepcopy(self.level_bindings),
            "_feature_order": [name for name, _ in self.feature_templates],
            "_groups": [],
            "_distribution": None,
            "_link": None,
            "_result": None,
            "_solver_result": None,
            "_linear_system_state": None,
            "_reporting_support_state": None,
            "_dm": None,
            "_fit_weights": None,
            "_fit_offset": None,
            "_fit_used_offset": False,
            "_fit_used_weights": False,
            "_fit_stats": None,
            "_runtime_canonical_state": None,
            "_nb_profile_result": None,
            "_tweedie_profile_result": None,
            "_last_fit_meta": None,
            "_monotone_repairs": {},
            "_prediction_plan": None,
            "_fast_prediction_state": None,
            "_fit_mu": None,
            "_fit_null_mu": None,
            "_fit_X_ref": None,
            "_fit_y_ref": None,
            "_fit_sample_weight_ref": None,
            "_fit_offset_ref": None,
            "_fit_data_guard": None,
            "_fit_geometry_guard": None,
            "_fit_metrics_cache": None,
            "_fit_metrics_cache_signature": None,
            "_summary_cache": None,
            "_interaction_specs": {
                name: copy.deepcopy(spec) for name, spec in self.interaction_templates
            },
            "_interaction_order": list(self.interaction_order),
            "_pending_interactions": tuple(self.interactions),
            "_config_revision": 0,
            "_fit_revision": 0,
            "_fit_state": None,
            "_selection_penalty_fitted": None,
            "_distribution_fitted": None,
            "_resolved_penalty": None,
        }
        work_model._config = type(self).capture(work_model)
        return work_model


@dataclass(frozen=True)
class ModelConfigPublication:
    """Complete constructor-state identities to install with a fitted revision."""

    config: ModelConfig
    revision: int
    penalty: object
    link: object
    family: object
    lambda2: object

    @classmethod
    def capture(cls, model) -> ModelConfigPublication:
        """Retain one model's already-owned constructor-state identities."""
        return cls(
            config=model._config,
            revision=int(model._config_revision),
            penalty=model._penalty_config,
            link=model._link_config,
            family=model._family_config,
            lambda2=model._lambda2_config,
        )

    def as_model_dict(self) -> dict[str, object]:
        """Return the coherent slots consumed by atomic publication."""
        return {
            "_config": self.config,
            "_config_revision": self.revision,
            "_penalty_config": self.penalty,
            "_link_config": self.link,
            "_family_config": self.family,
            "_lambda2_config": self.lambda2,
        }


@dataclass(frozen=True)
class FitState:
    """Authoritative identity and projections for one complete successful fit."""

    revision: int
    selection_penalty: float
    distribution: object
    projections: Mapping[str, object]
    retained: bool
    repair_revision: int = 0
    resolved_penalty: object | None = None
    resolved_lambda2: object | None = None


@dataclass(frozen=True)
class FitCandidate:
    """A fully prepared replacement dictionary ready for a no-fail install."""

    state: FitState
    prepared_model_dict: dict[str, Any]


@dataclass(frozen=True)
class FittedStateRevision:
    """Private coefficient-scale workspace for one atomic fitted-state revision."""

    target_model: Any
    model: Any
    revision: int
    repair_revision: int
    freeze_auxiliary_arrays: bool = True

    @classmethod
    def start(
        cls,
        model,
        *,
        increment: bool = True,
        freeze_auxiliary_arrays: bool = True,
    ) -> FittedStateRevision:
        """Clone fitted metadata without copying design or retained row buffers."""
        if (
            getattr(model, "_result", None) is None
            or getattr(model, "_solver_result", None) is None
        ):
            raise RuntimeError("a fitted-state revision requires a fitted model")

        work_model = copy.copy(model)
        work_model.__dict__ = dict(model.__dict__)
        result_copies: dict[int, object] = {}
        result_copy_memo: dict[int, object] = {}
        for result_name in ("_result", "_solver_result"):
            source_result = getattr(model, result_name)
            result_copy = result_copies.get(id(source_result))
            if result_copy is None:
                mutable_copy = getattr(source_result, "_mutable_copy", None)
                result_copy = (
                    mutable_copy(memo=result_copy_memo)
                    if mutable_copy is not None
                    else copy.deepcopy(source_result, result_copy_memo)
                )
                result_copies[id(source_result)] = result_copy
            setattr(work_model, result_name, result_copy)

        reml_result = getattr(model, "_reml_result", None)
        if reml_result is not None:
            nested_result = getattr(reml_result, "pirls_result", None)
            replacement = result_copies.get(id(nested_result))
            if replacement is not None:
                reml_copy = copy.copy(reml_result)
                reml_copy.pirls_result = replacement
                work_model._reml_result = reml_copy

        if not (
            _recorded_shift_reads_the_coefficients(work_model)
            and _intercepts_read_the_shift(work_model)
        ):
            # A model an editor revised before #447 kept the solver state of
            # its pre-edit coefficients, or moved both raw intercepts apart by
            # their own roundings: every revision starts from the published
            # predictor.
            _read_solver_state_from_public(work_model)

        current_state = getattr(model, "_fit_state", None)
        current_revision = int(getattr(model, "_fit_revision", 0))
        current_repair_revision = int(
            getattr(current_state, "repair_revision", 0) if current_state is not None else 0
        )
        step = int(bool(increment))
        return cls(
            target_model=model,
            model=work_model,
            revision=current_revision + step,
            repair_revision=current_repair_revision + step,
            freeze_auxiliary_arrays=bool(freeze_auxiliary_arrays),
        )

    def commit(self):
        """Validate and publish this complete revision through one dictionary swap."""
        if getattr(self.model, "_prediction_plan", None) is None:
            from superglm.model.base import freeze_prediction_plan

            freeze_prediction_plan(self.model)
        candidate = _capture_model_state(
            self.model,
            self.target_model,
            revision=self.revision,
            repair_revision=self.repair_revision,
            freeze_auxiliary_arrays=self.freeze_auxiliary_arrays,
        )
        _install_fit_state(self.target_model, candidate)
        return self.target_model


def canonical_intercept_shift(model, beta) -> float:
    """The public intercept's shift ``m' beta`` over the materialized public columns.

    ``m`` are the training means a materialized term's public columns lose
    (``runtime_canonicalize._apply_r_inv_centering``), so ``intercept_pub =
    intercept_solver + m' beta`` reads one predictor in both coordinates.
    """
    runtime_state = getattr(model, "_runtime_canonical_state", None)
    if not isinstance(runtime_state, dict):
        return 0.0
    shift = 0.0
    for term_state in runtime_state.get("terms", {}).values():
        if not term_state.get("applied_to_public_model", False):
            continue
        for group_state in term_state.get("groups", []):
            start, end = group_state["solver_slice"]
            means = np.asarray(group_state["column_means"], dtype=np.float64)
            shift += float(means @ np.asarray(beta[start:end], dtype=np.float64))
    return shift


def _recorded_shift_reads_the_coefficients(model) -> bool:
    """Whether the recorded shift ``m' beta`` is the one at the current ``beta``.

    Every revision records it again (``_read_solver_state_from_public``); an
    editor edit made before #447 left it, and the solver state it describes,
    at the pre-edit ``beta``.  The recorded shift is compared with the one at
    the current ``beta`` to the two sums' rounding, ``gamma_(k+1) sum |m|
    |beta|`` each over ``k`` terms in any order (Higham 2002, section 3.1).
    """
    state = getattr(model, "_runtime_canonical_state", None)
    public = getattr(model, "_result", None)
    if not isinstance(state, dict) or public is None:
        return True
    beta = np.asarray(public.beta, dtype=np.float64)
    shift = magnitude = 0.0
    count = 0
    for term_state in state.get("terms", {}).values():
        if not term_state.get("applied_to_public_model", False):
            continue
        for group_state in term_state.get("groups", []):
            start, end = group_state["solver_slice"]
            means = np.asarray(group_state["column_means"], dtype=np.float64)
            shift += float(means @ beta[start:end])
            magnitude += float(np.abs(means) @ np.abs(beta[start:end]))
            count += end - start
    unit = float(np.finfo(np.float64).eps) / 2.0
    # gamma_(k+1) / (1 - gamma_k) <= gamma_(2k+2) / 2: the factor 2 covers the
    # computed magnitude's own rounding and the roundings that form and apply it.
    gamma = 2 * (count + 1) * unit / (1.0 - 2 * (count + 1) * unit)
    return abs(float(state.get("intercept_shift", 0.0)) - shift) <= 2.0 * gamma * magnitude


def _intercepts_read_the_shift(model) -> bool:
    """Whether the raw intercepts read one predictor through the recorded shift ``h``.

    A fit publishes ``P = fl(S + h)`` and a revision reads ``S = fl(P - h)``
    (``_read_solver_state_from_public``).  In the second case ``fl(S + h) - P =
    d1 (P - h) + d2 (S + h)`` with ``|d1|, |d2| <= u``, so ``|P - fl(S + h)| <=
    u (|P| + |S| + 2 |h|) <= gamma_2 (|P| + |S| + |h|)`` (Higham 2002, section
    2.2); the first case is exact.  The gap is formed by one subtraction of
    nearby floats, exact by Sterbenz's lemma where it matters.  A bare
    ``1e-13`` checked this before #449, below one rounding of a solver
    intercept above about 900 (Claude's review of #448).
    """
    public = getattr(model, "_result", None)
    solver = getattr(model, "_solver_result", None)
    state = getattr(model, "_runtime_canonical_state", None)
    if public is None or solver is None or not isinstance(state, dict):
        return True
    shift = state.get("intercept_shift")
    if shift is None:
        return True
    public_intercept, solver_intercept = float(public.intercept), float(solver.intercept)
    unit = float(np.finfo(np.float64).eps) / 2.0
    gamma_2 = 2.0 * unit / (1.0 - 2.0 * unit)
    gap = abs(public_intercept - (solver_intercept + float(shift)))
    # Each magnitude is scaled before the sum, so the bound stays finite at the
    # binary64 limit, where a fit's own intercepts sit in a consistent state.
    # An inconsistent ``S + h`` past the limit leaves the gap infinite, and
    # ``inf <= inf`` would pass it (Codex's review of #453).
    bound = (
        gamma_2 * abs(public_intercept)
        + gamma_2 * abs(solver_intercept)
        + gamma_2 * abs(float(shift))
    )
    return math.isfinite(gap) and gap <= bound


def _carry_into_centred_pair(result, values) -> None:
    """Add ``fsum(values)`` to a result's centred intercept, keeping a compensated pair's error.

    ``math.fsum`` rounds the sum once (correctly under IEEE-754 binary64
    round-to-nearest, the Python documentation's condition); with a remainder
    ``alpha_lo`` the addition is a TwoSum whose error joins it (Knuth, TAOCP
    vol. 2, 4.2.2, Theorem B), so the pair moves by the sum to one rounding of
    the remainder.  A zero sum leaves the pair as it is, bit for bit.
    """
    alpha = getattr(result, "centred_intercept", None)
    if alpha is None or getattr(result, "state_center", None) is None:
        return
    move = math.fsum(values)
    if move == 0.0:
        return
    alpha_lo = getattr(result, "centred_intercept_lo", None)
    if alpha_lo is None:
        result.centred_intercept = float(alpha) + move
        return
    from superglm.solvers.mode_score import two_sum

    result.centred_intercept, error = two_sum(float(alpha), move)
    result.centred_intercept_lo = float(alpha_lo) + error


def move_public_intercept(model, delta: float) -> None:
    """Move the published predictor by ``delta``: its raw intercept and its centred pair alike.

    A revision's intercept change has to reach the pair the model predicts
    from.  Added to the raw intercepts alone it rounded away beside a numeric
    column at an offset of 1e16, whose raw intercept is about -2e15 (#447);
    a post-fit shape repair lost its profiled intercept the same way.  The
    solver's intercepts are read from the published ones when the revision is
    published (``publish_revised_coefficients``).
    """
    public = model._result
    public.intercept = float(public.intercept + delta)
    _carry_into_centred_pair(public, (float(delta),))


def _read_solver_state_from_public(model) -> None:
    """Read the solver's intercept, shift and centred pair from the published predictor.

    The public predictor is ``intercept_pub + X_pub beta`` raw and
    ``alpha_pub + (X_pub - 1 c_pub') beta`` centred, ``X_pub = X - 1 m'`` over
    the materialized columns (``runtime_canonicalize._fold_public_pair``).
    The solver's reads the same rows' values from ``intercept_pub - m' beta``
    raw and from ``alpha = alpha_pub - (m - c)' beta`` over the columns the
    public pair folds (``c_pub = 0``), centred: the inverse fold, a correctly
    rounded ``fsum`` and a TwoSum into the remainder.  Every other column is a
    numeric one, whose public shift ``m`` is zero, so its ``c_pub`` is ``c``
    and it adds nothing.  The solver's pair is evaluated as the public one is
    (``centred_sum_compensated``).  Left at its pre-revision
    values, the solver predictor, which a post-fit shape repair profiles its
    intercept from and metrics read on the training rows, read the old
    coefficients' shift ``m' beta_old`` after an editor edit, and the repair
    published the difference (#447).  With no published pair the public
    predictor is raw, and so is the solver's.
    """
    public = getattr(model, "_result", None)
    solver = getattr(model, "_solver_result", None)
    state = getattr(model, "_runtime_canonical_state", None)
    if public is None or solver is None or public is solver or not isinstance(state, dict):
        return
    # The canonical state is shared with the model the revision started from.
    state = copy.deepcopy(state)
    model._runtime_canonical_state = state
    beta = np.asarray(public.beta, dtype=np.float64)
    shift = canonical_intercept_shift(model, beta)
    state["intercept_shift"] = float(shift)
    diagnostics = state.get("diagnostics")
    if isinstance(diagnostics, dict):
        diagnostics["intercept_shift"] = float(shift)
        diagnostics["coefficients_revised"] = True
    for term_state in state.get("terms", {}).values():
        term_shift = 0.0
        for group_state in term_state.get("groups", []):
            start, end = group_state["solver_slice"]
            means = np.asarray(group_state["column_means"], dtype=np.float64)
            term_shift += float(means @ beta[start:end])
        term_state["intercept_shift"] = term_shift
    solver.intercept = float(public.intercept) - shift

    centre = getattr(solver, "state_center", None)
    if getattr(solver, "centred_intercept", None) is None or centre is None:
        return
    alpha_public = getattr(public, "centred_intercept", None)
    public_centre = getattr(public, "state_center", None)
    if alpha_public is None or public_centre is None:
        for field_name in ("centred_intercept", "state_center", "centred_intercept_lo"):
            setattr(solver, field_name, None)
        solver.centred_sum_compensated = False
        return
    solver.centred_sum_compensated = bool(getattr(public, "centred_sum_compensated", False))
    from superglm.model.runtime_canonicalize import _public_column_shifts
    from superglm.solvers.mode_score import two_sum

    centre = np.asarray(centre, dtype=np.float64)
    folded = np.asarray(public_centre, dtype=np.float64) == 0.0
    shifts = _public_column_shifts(state.get("terms", {}), centre.size)
    fold = math.fsum((shifts[folded] - centre[folded]) * beta[folded])
    alpha_public_lo = getattr(public, "centred_intercept_lo", None)
    if alpha_public_lo is None:
        solver.centred_intercept = float(alpha_public) - fold
        solver.centred_intercept_lo = None
        return
    solver.centred_intercept, error = two_sum(float(alpha_public), -fold)
    solver.centred_intercept_lo = float(alpha_public_lo) + error


def _exact_rounded_sum(weights: np.ndarray, values: np.ndarray) -> float:
    """``sum w v`` from exact products, rounded once (``mode_score._exact_sum``)."""
    from superglm.solvers._exact_sums import native_operand
    from superglm.solvers.mode_score import _exact_sum

    total, exponent, exact = _exact_sum(native_operand(weights), native_operand(values), 0.0, 1)
    if not exact:  # pragma: no cover - finite operands always sum
        return math.inf
    try:
        return math.ldexp(total, exponent)
    except OverflowError:
        # The exact sum is finite but past the binary64 range (a year column
        # edited to a slope of 1e306: Claude's review of #453).
        return math.copysign(math.inf, total)


def _carry_revised_columns(model, beta_before) -> None:
    """Carry the change of each centred column a revision moved into the pair, exactly.

    A revision writes raw coordinates: a numeric column's new slope pivots at
    ``x = 0``, so about its fitted centre ``c`` the column contributes ``(x -
    c) beta`` and the pair takes ``K = sum c beta - c beta_before``.  ``K`` is
    formed from exact products, ``c beta`` and ``-c beta_before`` and never
    their difference (which -1e308 to 1e308 overflows), and split into its
    correctly rounded value and the rounded remainder, so the pair holds it to
    ``u^2``.  The centres stay where the fit put them.  A zero-centre column
    contributes ``x beta`` already and is skipped.

    ``alpha`` then holds ``c dbeta``, which a row's ``(x - c) beta`` can cancel:
    ``1e16 + 2`` beside ``-1e16`` at ``x = 0`` in Sol's review of #448.  The
    result is marked ``centred_sum_compensated`` and evaluated as one
    compensated sum with each column's exact centred product
    (``mode_score.linear_predictor``, ``model.base.EtaSum``), so the remainder
    survives the cancellation.  Moving the centre instead (0.36.0, ``c' = c
    beta_before / beta``) lost a cancellation between two edited columns
    (#449).  A pair with no remainder gains one here, whatever the family.
    """
    public = model._result
    centre = getattr(public, "state_center", None)
    alpha = getattr(public, "centred_intercept", None)
    if centre is None or alpha is None:
        return
    centre = np.asarray(centre, dtype=np.float64)
    before = np.asarray(beta_before, dtype=np.float64)
    beta = np.asarray(public.beta, dtype=np.float64)
    moved = np.flatnonzero((centre != 0.0) & (beta != before))
    if not moved.size:
        return
    from superglm.solvers.mode_score import two_sum

    weights = np.concatenate([centre[moved], -centre[moved]])
    values = np.concatenate([beta[moved], before[moved]])
    high = _exact_rounded_sum(weights, values)
    if math.isfinite(high):
        low = _exact_rounded_sum(np.append(weights, -1.0), np.append(values, high))
        alpha_high, error = two_sum(float(alpha), high)
        alpha_low = float(getattr(public, "centred_intercept_lo", None) or 0.0) + (error + low)
        if math.isfinite(alpha_high) and math.isfinite(alpha_low):
            public.centred_intercept = alpha_high
            public.centred_intercept_lo = alpha_low
            public.centred_sum_compensated = True
            return
    # A change past the binary64 range cannot be carried.  The predictor falls
    # back to the raw intercept, as 31544462's editor left every slope edit.  A
    # row whose ``x beta`` overflows is not finite on any version, and the
    # edit's refreshed deviance refuses it as v0.36.0 does.
    for field_name in ("centred_intercept", "state_center", "centred_intercept_lo"):
        setattr(public, field_name, None)
    public.centred_sum_compensated = False


def publish_revised_coefficients(model, beta_before) -> None:
    """Publish a revision of the fitted coefficients as one predictor in every coordinate.

    A revision (an editor edit, a post-fit shape repair) writes ``beta`` into
    both results and moves the public intercept through
    ``move_public_intercept``; the raw public predictor ``intercept_pub +
    X_pub beta`` states what it means.  The change of a column the pair keeps
    centred (a numeric one) that the revision moved is carried into the pair
    exactly (``_carry_revised_columns``), so that ``alpha_pub + (X_pub - 1
    c_pub') beta`` predicts the same rows; every other column carries no centre
    and nothing moves.  The solver state is then read from the published one
    (``_read_solver_state_from_public``) and the fitted mode's identity is
    cleared.  The pair is carried, never dropped: clearing it dropped every
    dense column's centring (Sol's review of #445, P2), and keeping the
    solver's pre-revision pair beside it left the solver predictor at the old
    coefficients (#447).
    """
    _carry_revised_columns(model, beta_before)
    _read_solver_state_from_public(model)

    updated: set[int] = set()
    for result_name in ("_result", "_solver_result"):
        result = getattr(model, result_name, None)
        if result is None or id(result) in updated:
            continue
        for field_name, value in (
            ("state_id", None),
            ("evaluation_id", None),
            ("log_det_H", None),
            ("reml_hessian_rank", None),
            ("converged", False),
            ("termination_reason", "coefficients_revised"),
            ("scop_inference", None),
            ("scop_geometry", None),
        ):
            if hasattr(result, field_name):
                setattr(result, field_name, value)
        updated.add(id(result))

    reml_result = getattr(model, "_reml_result", None)
    if reml_result is not None:
        reml_result.objective = None
        reml_result.converged = False
        reml_result.scop_states = None
        reml_result.curvature_source = None
        reml_result.termination_reason = "coefficients_revised"

    # Profile likelihoods and confidence intervals are functions of the old
    # fitted mean, so they cannot survive an arbitrary coefficient revision.
    model._linear_system_state = None
    model._reporting_support_state = None
    model._nb_profile_result = None
    model._tweedie_profile_result = None


_FIT_PROJECTION_NAMES = (
    "_result",
    "_solver_result",
    "_linear_system_state",
    "_reporting_support_state",
    "_dm",
    "_groups",
    "_specs",
    "_feature_order",
    "_interaction_specs",
    "_interaction_order",
    "_distribution",
    "_link",
    "_fit_weights",
    "_fit_offset",
    "_fit_used_offset",
    "_fit_used_weights",
    "_fit_stats",
    "_runtime_canonical_state",
    "_nb_profile_result",
    "_tweedie_profile_result",
    "_last_fit_meta",
    "_prediction_plan",
    "_fast_prediction_state",
    "_fit_mu",
    "_fit_null_mu",
    "_fit_X_ref",
    "_fit_y_ref",
    "_fit_sample_weight_ref",
    "_fit_offset_ref",
    "_fit_data_guard",
    "_fit_geometry_guard",
    "_fit_metrics_cache",
    "_fit_metrics_cache_signature",
    "_summary_cache",
    "_reml_lambdas",
    "_reml_penalties",
    "_reml_result",
    "_reml_profile",
)


def _validate_workspace_result(work_model) -> float:
    """Validate the minimum structural and finite invariants before publication."""
    result = work_model._result
    solver_result = work_model._solver_result
    if result is None or solver_result is None:
        raise RuntimeError("fit candidate is missing a finalized solver result")
    beta = np.asarray(result.beta)
    solver_beta = np.asarray(solver_result.beta)
    if beta.ndim != 1 or solver_beta.ndim != 1 or beta.shape != solver_beta.shape:
        raise RuntimeError("fit candidate coefficient dimensions are inconsistent")
    if work_model._dm is not None and beta.shape != (work_model._dm.p,):
        raise RuntimeError("fit candidate coefficients do not match its design matrix")
    expected_p = sum(group.size for group in work_model._groups)
    if beta.shape != (expected_p,):
        raise RuntimeError("fit candidate coefficients do not match its fitted groups")
    if not np.all(np.isfinite(beta)) or not np.all(np.isfinite(solver_beta)):
        raise RuntimeError("fit candidate coefficients must be finite")
    if not np.array_equal(beta, solver_beta):
        raise RuntimeError("fit candidate public and solver coefficients are inconsistent")
    scalars = (
        result.intercept,
        result.deviance,
        result.phi,
        result.effective_df,
        solver_result.intercept,
        solver_result.deviance,
        solver_result.phi,
        solver_result.effective_df,
    )
    if not np.all(np.isfinite(np.asarray(scalars, dtype=np.float64))):
        raise RuntimeError("fit candidate scalar results must be finite")
    public_solver_scalars = (
        (result.deviance, solver_result.deviance),
        (result.phi, solver_result.phi),
        (result.effective_df, solver_result.effective_df),
    )
    if any(float(public) != float(solver) for public, solver in public_solver_scalars) or (
        int(result.n_iter) != int(solver_result.n_iter)
        or bool(result.converged) is not bool(solver_result.converged)
    ):
        raise RuntimeError("fit candidate public and solver scalar results are inconsistent")
    if work_model._distribution is None or work_model._link is None:
        raise RuntimeError("fit candidate is missing its resolved distribution or link")
    if work_model._runtime_canonical_state is None or work_model._prediction_plan is None:
        raise RuntimeError("fit candidate was not canonically finalized")
    intercept_shift = work_model._runtime_canonical_state.get("intercept_shift")
    if intercept_shift is None or not np.isfinite(intercept_shift):
        raise RuntimeError("fit candidate is missing its canonical intercept shift")
    if not _intercepts_read_the_shift(work_model):
        raise RuntimeError("fit candidate canonical intercept relation is inconsistent")
    reml_result = getattr(work_model, "_reml_result", None)
    if reml_result is not None and getattr(reml_result, "pirls_result", None) is not solver_result:
        raise RuntimeError("fit candidate REML result does not reference its solver result")
    selection_penalty = getattr(fitted_penalty(work_model), "lambda1", None)
    if selection_penalty is None or not np.isfinite(selection_penalty):
        raise RuntimeError("fit candidate is missing its resolved selection penalty")
    return float(selection_penalty)


def _freeze_array(value: object) -> None:
    if isinstance(value, np.ndarray):
        value.setflags(write=False)


def _freeze_candidate_arrays(work_model, *, auxiliary: bool = True) -> None:
    """Freeze attempt-owned fitted arrays without traversing the design matrix."""
    published_results: set[int] = set()
    publication_memo: dict[int, object] = {}
    for result_name in ("_result", "_solver_result"):
        result = getattr(work_model, result_name, None)
        if result is None or id(result) in published_results:
            continue
        publisher = getattr(result, "_publish", None)
        if publisher is None:
            _freeze_array(getattr(result, "beta", None))
            for row in getattr(result, "iteration_log", None) or ():
                _freeze_array(getattr(row, "top_w_indices", None))
                _freeze_array(getattr(row, "bottom_w_indices", None))
        else:
            publisher(publication_memo)
        published_results.add(id(result))

    if not auxiliary:
        return

    covariance = work_model.__dict__.get("_coef_covariance")
    if isinstance(covariance, tuple) and covariance:
        _freeze_array(covariance[0])

    inference = work_model.__dict__.get("_fit_inference_info")
    if isinstance(inference, dict):
        for value in inference.values():
            _freeze_array(value)

    for name in ("_fit_weights", "_fit_offset", "_fit_mu", "_fit_null_mu"):
        _freeze_array(getattr(work_model, name, None))


def _publish_workspace_extension_state(work_model, public_model, prepared) -> None:
    """Detach tracked extension state and rebind workspace self-references."""
    from superglm.model.fit_workspace import _SUBCLASS_STATE_NAMES

    tracked_names = tuple(getattr(work_model, _SUBCLASS_STATE_NAMES, ()))
    if not tracked_names:
        return

    # A second, shared deepcopy makes the published extension graph independent
    # of the private attempt while preserving aliases. Self-references and
    # stored bound methods must resolve to the durable public model, not to the
    # workspace whose dictionary is about to be transferred.
    memo = {
        id(work_model): public_model,
        id(public_model): public_model,
    }
    tracked = set(tracked_names)
    for name, source_value in work_model.__dict__.items():
        if name in tracked:
            continue
        published_value = prepared.get(name)
        if source_value is published_value:
            memo[id(source_value)] = published_value

    # Configuration identities are intentionally restored from the caller at
    # publication. Preserve direct aliases from extension state to those
    # model-owned objects without mapping scalar singleton identities.
    scalar_types = (type(None), bool, int, float, complex, str, bytes)
    for name in (
        "_config",
        "_penalty_config",
        "_link_config",
        "_family_config",
        "_lambda2_config",
    ):
        source_value = work_model.__dict__.get(name)
        published_value = prepared.get(name)
        if not isinstance(source_value, scalar_types) and published_value is not None:
            memo[id(source_value)] = published_value

    for name in tracked_names:
        if name not in work_model.__dict__:
            raise RuntimeError(f"tracked subclass fit state {name!r} is missing")
        try:
            prepared[name] = copy.deepcopy(work_model.__dict__[name], memo)
        except Exception as exc:  # pragma: no cover - depends on extension object
            raise TypeError(
                f"subclass fit state {name!r} must support deepcopy for transactional fitting"
            ) from exc


def _capture_model_state(
    work_model,
    public_model,
    *,
    revision: int,
    repair_revision: int,
    freeze_auxiliary_arrays: bool,
    config_publication: ModelConfigPublication | None = None,
) -> FitCandidate:
    """Build a validated candidate from a complete private fitted model."""
    selection_penalty = _validate_workspace_result(work_model)
    _freeze_candidate_arrays(work_model, auxiliary=freeze_auxiliary_arrays)
    projections = {
        name: getattr(work_model, name)
        for name in _FIT_PROJECTION_NAMES
        if hasattr(work_model, name)
    }
    resolved_penalty = fitted_penalty(work_model)
    # A fitted-state revision begins from an installed result whose resolved
    # smoothing values remain authoritative even when constructor intent was
    # staged for the *next* fit.  Fresh fit workspaces have no installed state
    # and therefore resolve from their own REML/configured values as before.
    installed_state = getattr(work_model, "_fit_state", None)
    if installed_state is not None and installed_state.resolved_lambda2 is not None:
        resolved_lambda2 = fitted_lambda2(work_model)
    else:
        optimized_reml_lambdas = getattr(work_model, "_reml_lambdas", None)
        resolved_lambda2 = (
            optimized_reml_lambdas
            if optimized_reml_lambdas is not None
            else configured_lambda2(work_model)
        )
    state = FitState(
        revision=int(revision),
        selection_penalty=selection_penalty,
        distribution=work_model._distribution,
        projections=FrozenMapping(projections),
        retained=bool(work_model._retain_fit_state),
        repair_revision=int(repair_revision),
        resolved_penalty=resolved_penalty,
        resolved_lambda2=copy.deepcopy(resolved_lambda2),
    )

    # Shallow transfer is intentional: every large buffer was created by this
    # attempt and ownership moves to the installed model without duplication.
    prepared = dict(work_model.__dict__)
    publication = (
        ModelConfigPublication.capture(public_model)
        if config_publication is None
        else config_publication
    )
    prepared.update(publication.as_model_dict())
    prepared["_resolved_penalty"] = resolved_penalty
    prepared["_selection_penalty_fitted"] = selection_penalty
    prepared["_distribution_fitted"] = work_model._distribution
    prepared["_fit_revision"] = int(revision)
    prepared["_fit_state"] = state
    _publish_workspace_extension_state(work_model, public_model, prepared)
    return FitCandidate(state=state, prepared_model_dict=prepared)


def capture_fit_state(
    workspace,
    public_model,
    *,
    revision: int,
    config_publication: ModelConfigPublication | None = None,
) -> FitCandidate:
    """Transfer one complete workspace into a candidate without row-scale copies."""
    design = getattr(workspace.model, "_dm", None)
    release_raw_splines = getattr(design, "release_raw_spline_tabmat_plan", None)
    if release_raw_splines is not None:
        # The raw-basis Tabmat plan accelerates all coefficient/REML candidates
        # in this workspace, but duplicates the durable sparse spline basis.
        # Drop that rebuildable cache before ownership moves to the public fit.
        release_raw_splines()
    if design is not None:
        from superglm.solvers.centered_system import release_dense_split

        # Beside a dense column the bounded half of the design carries its own
        # copies of those caches (``centered_system._DENSE_SPLITS``); they go too.
        release_dense_split(design)
    return _capture_model_state(
        workspace.model,
        public_model,
        revision=revision,
        repair_revision=0,
        freeze_auxiliary_arrays=True,
        config_publication=config_publication,
    )


def _install_fit_state(model, candidate: FitCandidate) -> None:
    """Publish a prevalidated candidate through one allocation-free swap."""
    model.__dict__ = candidate.prepared_model_dict
