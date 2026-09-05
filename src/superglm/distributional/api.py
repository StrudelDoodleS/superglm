"""Public model façade for auditable multi-predictor distributional fitting."""

from __future__ import annotations

import hashlib
import hmac
import importlib
import json
import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm._blas_threads import solver_blas_threads
from superglm._frame import EagerFrame, FrameLike, as_eager_frame
from superglm.diagnostics.fit_report import FitDiagnosticReport
from superglm.distributional.checks.binned import (
    BinnedCheck,
    BinnedCheck2D,
    binned_check,
    binned_check_2d,
)
from superglm.distributional.checks.calibration import (
    ActualExpected,
    CalibrationPayload,
    actual_expected_check,
    calibration_payload,
)
from superglm.distributional.checks.compare import Comparison, ScoreName, compare_models
from superglm.distributional.checks.pit import PITPayload, pit_payload
from superglm.distributional.checks.qq import QQPayload, qq_payload
from superglm.distributional.checks.scores import score_table
from superglm.distributional.checks.worm import WormPayload, worm_payload
from superglm.distributional.family import (
    DistributionalFamily,
    ExpectedInformationFamily,
    FitFailureDiagnosingFamily,
    ParameterSpec,
    _validated_complete_fit_configuration,
    validate_family,
)
from superglm.distributional.fit_diagnostics import diagnose_distributional_fit
from superglm.distributional.model import (
    DenseDistributionalModel,
    _clone_predictor_templates,
    _take_unvalidated_offsets,
    _unvalidated_offset_shapes,
    fit_dense_distributional,
)
from superglm.distributional.posterior import (
    CovarianceKind,
    PosteriorDraws,
    Quantity,
    posterior_bounds,
    posterior_covariance,
    posterior_draws,
    posterior_predictive,
)
from superglm.distributional.predictor import Predictor
from superglm.distributional.residuals import ResidualKind, ResidualSet, compute_residuals
from superglm.distributional.result import (
    DenseSolverConfig,
    DistributionalEFSConfig,
    DistributionalFitResult,
    EFSConvergenceReason,
)
from superglm.distributional.separation import SeparationPolicy, validate_separation_policy
from superglm.distributional.serialization import (
    DistributionalSerializationError,
    deserialize_distributional_model,
    serialize_distributional_model,
)
from superglm.distributional.solver.curvature import RepeatedCurvatureIndefinitenessError
from superglm.distributional.surfaces import (
    DensityFan,
    Portfolio,
    RiskCurves,
    Spread,
    density_fan,
    parameter_spread,
    portfolio,
    risk_curves,
)
from superglm.distributional.telemetry import CurvatureTelemetry
from superglm.distributional.terms import (
    ParameterTermEffect,
    TermTest,
    _compiled_spec,
    _prepare_term_effect,
    _term_effect_from_covariance,
    summary_table,
    term_effect,
    term_test,
)
from superglm.distributional.timing import FitPhaseRecorder, FitPhaseSnapshot
from superglm.distributional.weights import (
    LegacyPowerWeightArtifactError,
    UnsupportedLikelihoodContractError,
    WeightContract,
    resolve_likelihood_weights,
)
from superglm.links import Link, resolve_link


@dataclass(frozen=True)
class SuperLSSTrainingTelemetry:
    """Immutable fit-scale telemetry available without retained training arrays."""

    model_type: str
    family: str
    parameter_names: tuple[str, ...]
    curvature_policy: str
    n_observations: int
    predictor_dimensions: Mapping[str, int]
    total_dimension: int
    curvature_channels: int
    discrete: bool
    n_bins: int | Mapping[str, int] | None
    inner_iterations: int
    smoothing_iterations: int
    smoothing_convergence_reason: EFSConvergenceReason | None
    smoothing_certified: bool | None
    smoothing_unresolved_upper_bound: tuple[str, ...] | None
    converged: bool
    rank: int
    backtracking_steps: int
    curvature: CurvatureTelemetry

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "predictor_dimensions",
            MappingProxyType(dict(self.predictor_dimensions)),
        )
        if isinstance(self.n_bins, Mapping):
            object.__setattr__(self, "n_bins", MappingProxyType(dict(self.n_bins)))


@dataclass(frozen=True)
class _LinkDefaults:
    default_link: str | Link


_PUBLIC_ARTIFACT_TYPE = "superglm.SuperLSS"

# Versions the ``public_api`` block that :meth:`SuperLSS.to_bytes` wraps around
# the dense-model envelope -- its ``artifact_type`` / ``schema_version`` /
# ``config`` / ``sha256`` keys, and nothing below them.  It is deliberately
# independent of ``serialization.SCHEMA_VERSION``, which versions the envelope
# underneath.  Version 2 is the first released wrapper schema.  Version 1 was
# an internal pre-contract shape and is explicitly refused because it cannot
# prove weight semantics.  Both numbers are written into every artifact, so
# whichever layer refuses a stale artifact names its own version.
# ``test_public_schema_version_is_independent_of_the_envelope_version`` pins the
# relationship, including the block's key set.
_PUBLIC_SCHEMA_VERSION = "2.0.0"


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise DistributionalSerializationError(
            "SuperLSS artifact metadata must be finite JSON"
        ) from exc


def _public_config_digest(
    config: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> str:
    payload = artifact.get("payload")
    fitted_digest = payload.get("sha256") if isinstance(payload, Mapping) else None
    if not isinstance(fitted_digest, str) or len(fitted_digest) != 64:
        raise DistributionalSerializationError("fitted-state payload digest is invalid")
    digest = hashlib.sha256()
    digest.update(fitted_digest.lower().encode("ascii"))
    digest.update(b"\n")
    digest.update(_canonical_json(config))
    return digest.hexdigest()


def _parse_public_artifact(
    serialized: bytes | bytearray | memoryview | str,
) -> tuple[
    Mapping[str, Any],
    WeightContract,
    bool,
    int | Mapping[str, int],
    SeparationPolicy,
]:
    if isinstance(serialized, str):
        text = serialized
    elif isinstance(serialized, bytes | bytearray | memoryview):
        try:
            text = bytes(serialized).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DistributionalSerializationError("SuperLSS artifact must be UTF-8 JSON") from exc
    else:
        raise TypeError("serialized artifact must be bytes or str")

    try:
        artifact = json.loads(text)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise DistributionalSerializationError("SuperLSS artifact is not valid JSON") from exc
    if not isinstance(artifact, Mapping):
        raise DistributionalSerializationError("SuperLSS artifact root must be an object")
    public = artifact.get("public_api")
    if not isinstance(public, Mapping):
        raise DistributionalSerializationError("SuperLSS public metadata is missing")
    if public.get("artifact_type") != _PUBLIC_ARTIFACT_TYPE:
        raise DistributionalSerializationError("artifact type is not SuperLSS")
    public_version = public.get("schema_version")
    if not isinstance(public_version, str):
        raise DistributionalSerializationError(
            "SuperLSS metadata schema_version must be a semantic version"
        )
    version_parts = public_version.split(".")
    if len(version_parts) != 3 or any(not part.isdigit() for part in version_parts):
        raise DistributionalSerializationError(
            "SuperLSS metadata schema_version must be a semantic version"
        )

    config = public.get("config")
    if not isinstance(config, Mapping):
        raise DistributionalSerializationError("SuperLSS public configuration is incomplete")
    digest = public.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise DistributionalSerializationError("SuperLSS public configuration digest is invalid")
    expected = _public_config_digest(config, artifact)
    if not hmac.compare_digest(digest.lower(), expected):
        raise DistributionalSerializationError(
            "SuperLSS public configuration digest does not match"
        )
    public_major = int(version_parts[0])
    current_major = int(_PUBLIC_SCHEMA_VERSION.split(".", 1)[0])
    if public_major == 1:
        if set(config) != {"discrete", "n_bins"}:
            raise DistributionalSerializationError(
                "legacy SuperLSS public configuration is corrupt"
            )
        raise LegacyPowerWeightArtifactError(
            f"legacy pre-contract SuperLSS metadata version {public_version} cannot "
            "establish likelihood-weight semantics; refit and regenerate the artifact"
        )
    if public_major != current_major:
        raise DistributionalSerializationError(
            f"artifact SuperLSS metadata version {public_version!r} is unreadable by this "
            f"build, which reads and writes version {_PUBLIC_SCHEMA_VERSION!r}"
        )
    if set(config) != {"discrete", "n_bins", "separation", "weight_contract"}:
        raise DistributionalSerializationError("SuperLSS public configuration is incomplete")
    contract_config = config["weight_contract"]
    if (
        not isinstance(contract_config, Mapping)
        or set(contract_config) != {"schema_version", "semantics"}
        or type(contract_config["schema_version"]) is not int
        or contract_config["schema_version"] != 1
    ):
        raise DistributionalSerializationError(
            "SuperLSS likelihood-weight contract configuration is invalid"
        )
    try:
        contract = WeightContract(semantics=contract_config["semantics"])
    except (TypeError, UnsupportedLikelihoodContractError) as exc:
        raise DistributionalSerializationError(
            "SuperLSS likelihood-weight contract configuration is invalid"
        ) from exc
    discrete = config["discrete"]
    if not isinstance(discrete, bool):
        raise DistributionalSerializationError("SuperLSS discrete configuration must be bool")
    try:
        n_bins = _owned_n_bins(config["n_bins"])
    except (TypeError, ValueError) as exc:
        raise DistributionalSerializationError("SuperLSS n_bins configuration is invalid") from exc
    try:
        separation = validate_separation_policy(config["separation"])
    except ValueError as exc:
        raise DistributionalSerializationError(
            "SuperLSS separation configuration is invalid"
        ) from exc
    return artifact, contract, discrete, n_bins, separation


def _predictor_error(
    parameters: tuple[ParameterSpec, ...],
    predictors: tuple[Predictor, ...],
) -> None:
    expected = tuple(parameter.name for parameter in parameters)
    actual = tuple(predictor.name for predictor in predictors)
    duplicates = sorted({name for name in actual if actual.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate predictor name: {', '.join(duplicates)}")
    unknown = tuple(name for name in actual if name not in expected)
    if unknown:
        raise ValueError(f"unknown predictor name: {', '.join(unknown)}")
    missing = tuple(name for name in expected if name not in actual)
    if missing:
        raise ValueError(f"missing predictor name: {', '.join(missing)}")
    if actual != expected:
        raise ValueError(f"predictor order must match family order {expected}; got {actual}")


def _validate_link_support(parameter: ParameterSpec, predictor: Predictor) -> None:
    try:
        link = (
            parameter.default_link
            if predictor.link is None and isinstance(parameter.default_link, Link)
            else resolve_link(
                predictor.link,
                cast(Any, _LinkDefaults(parameter.default_link)),
            )
        )
        eta = np.array([-20.0, -2.0, 0.0, 2.0, 20.0])
        with np.errstate(all="ignore"):
            values = np.asarray(link.inverse(eta), dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"link for predictor {predictor.name!r} is incompatible") from exc
    if values.shape != eta.shape or not np.all(parameter.support.contains(values)):
        raise ValueError(f"link for predictor {predictor.name!r} is incompatible with its support")


def _owned_predictors(
    family: DistributionalFamily,
    predictors: Sequence[Predictor],
) -> tuple[Predictor, ...]:
    parameters = validate_family(family)
    _validated_complete_fit_configuration(family)
    if isinstance(predictors, str | bytes) or not isinstance(predictors, Sequence):
        raise TypeError("predictors must be an ordered sequence of Predictor values")
    values = tuple(predictors)
    if not values or not all(isinstance(predictor, Predictor) for predictor in values):
        raise TypeError("predictors must contain only Predictor values")
    _predictor_error(parameters, values)
    for parameter, predictor in zip(parameters, values, strict=True):
        _validate_link_support(parameter, predictor)
    return _clone_predictor_templates(values)


def _coefficient_curvature(
    family: DistributionalFamily,
    requested: object,
) -> Literal["fisher", "observed"]:
    """Return the coefficient-curvature policy for the dense solve.

    Every family is solved with the observed Hessian unless Fisher scoring is
    requested, and a request needs the expected-information capability.  The
    fallback is unchanged and lives in the solver: a materially indefinite
    terminal penalized observed Hessian falls back to penalized expected
    information when the family supplies it and is refused otherwise.  The
    chunked route keeps its own requirement in ``solver/solver.py``.
    """
    if requested not in ("observed", "fisher"):
        raise ValueError("coefficient_curvature must be 'observed' or 'fisher'")
    if requested == "fisher" and not isinstance(family, ExpectedInformationFamily):
        raise ValueError(
            "coefficient_curvature='fisher' requires a family with expected information capability"
        )
    return cast(Literal["fisher", "observed"], requested)


def _diagnosed_repeated_curvature_failure(
    family: DistributionalFamily,
    y: NDArray,
    *,
    sample_weight: NDArray | None,
    weight_contract: WeightContract,
    failure: RepeatedCurvatureIndefinitenessError,
) -> Exception | None:
    if type(failure) is not RepeatedCurvatureIndefinitenessError or not isinstance(
        family, FitFailureDiagnosingFamily
    ):
        return None
    try:
        response = np.asarray(y)
        if response.ndim != 1:
            return None
        weights = resolve_likelihood_weights(
            sample_weight,
            n_observations=len(response),
            contract=weight_contract,
        )
        candidate = family.diagnose_repeated_curvature_failure(response, weights)
    except Exception:
        return None
    return candidate if isinstance(candidate, Exception) else None


def _positive_n_bins(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a positive integer") from exc
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(result)


def _owned_n_bins(value: int | Mapping[str, int]) -> int | Mapping[str, int]:
    if isinstance(value, Mapping):
        result: dict[str, int] = {}
        for name, count in value.items():
            if not isinstance(name, str) or not name:
                raise ValueError("n_bins keys must be non-empty feature names")
            result[name] = _positive_n_bins(count, name=f"n_bins[{name!r}]")
        return MappingProxyType(result)
    return _positive_n_bins(value, name="n_bins")


def _prediction_index(X: FrameLike | EagerFrame, n_observations: int) -> pd.Index:
    if isinstance(X, pd.DataFrame):
        return X.index.copy()
    if isinstance(X, EagerFrame) and isinstance(X.native, pd.DataFrame):
        return X.native.index.copy()
    return pd.RangeIndex(n_observations)


def _phase_delta(
    after: FitPhaseSnapshot,
    before: FitPhaseSnapshot,
) -> FitPhaseSnapshot:
    """The phase timing of one fit, isolated from whatever the recorder held before it."""
    return FitPhaseSnapshot(
        seconds={name: after.seconds[name] - before.seconds[name] for name in after.seconds},
        counts={name: after.counts[name] - before.counts[name] for name in after.counts},
    )


#: Renderer module and figure names of each engine, imported only when asked
#: for, so ``plotly`` is never a requirement of importing this module.
_RENDERERS = {
    "matplotlib": (
        "superglm.plotting.distributional",
        {"diagnostics": "plot_diagnostics_figure", "term_grid": "plot_term_grid"},
    ),
    "plotly": (
        "superglm.plotting.distributional_plotly",
        {"diagnostics": "plotly_diagnostics_figure", "term_grid": "plotly_term_grid"},
    ),
}

#: Payload names :meth:`SuperLSS.plot_data` knows, each the JSON form of one builder.
_PLOT_DATA_KINDS = (
    "qq",
    "worm",
    "pit",
    "binned",
    "actual_expected",
    "calibration",
    "scores",
    "comparison",
    "term",
    "risk_curves",
    "density_fan",
    "spread",
    "portfolio",
)


def _renderer(engine: str, figure: str) -> Any:
    """Return one renderer of one engine, importing that engine only now."""
    if engine not in _RENDERERS:
        raise ValueError(f"engine must be 'matplotlib' or 'plotly', not {engine!r}")
    module_name, names = _RENDERERS[engine]
    return getattr(importlib.import_module(module_name), names[figure])


def _json_number(value: Any) -> float | None:
    """One number of a JSON payload: a float, or ``None`` where it is not finite."""
    number = float(value)
    return number if math.isfinite(number) else None


class SuperLSS:
    """Certification-aware distributional model with ordered family predictors."""

    def __init__(
        self,
        *,
        family: DistributionalFamily,
        predictors: Sequence[Predictor],
        weight_semantics: Literal["prior", "frequency"] = "prior",
        discrete: bool = False,
        n_bins: int | Mapping[str, int] = 256,
        separation: Literal["warn", "error", "ignore"] = "warn",
        coefficient_curvature: Literal["observed", "fisher"] = "observed",
    ) -> None:
        if not isinstance(discrete, bool):
            raise TypeError("discrete must be bool")
        self._separation = validate_separation_policy(separation)
        if discrete:
            raise NotImplementedError(
                "Discrete SuperLSS fitting is not implemented yet. Use discrete=False; "
                "discrete fitting remains available for scalar SuperGLM models."
            )
        self._family = family
        self._predictors = _owned_predictors(family, predictors)
        self._coefficient_curvature = _coefficient_curvature(family, coefficient_curvature)
        self._weight_contract = WeightContract(semantics=weight_semantics)
        self._discrete = discrete
        self._n_bins = _owned_n_bins(n_bins)
        self._model: DenseDistributionalModel | None = None
        self._fit_phase_snapshot: FitPhaseSnapshot | None = None
        # The term grids sweep one covariate over its training range with the
        # others held at their training centre, so they need the frame the fit
        # saw.  It is kept by reference, not copied, and serialization does not
        # carry it: a restored model is given one through ``X_train=``.
        self._training_frame: FrameLike | EagerFrame | None = None

    @property
    def family(self) -> DistributionalFamily:
        return self._family

    @property
    def weight_semantics(self) -> str:
        """Return what a ``sample_weight`` entry says about a row.

        ``"prior"`` (the default, as on the scalar path) reads a weight as how
        precisely that row was measured; ``"frequency"`` reads it as how many
        identical rows it stands for.  The contract decides whether learned
        geometry -- knot placement, discretized bins -- follows weight mass or
        physical rows.
        """
        return self._weight_contract.semantics

    @property
    def separation(self) -> str:
        """Return the build-time policy for separated categorical cells.

        A ``Categorical`` level or ``CategoricalInteraction`` cell that carries
        exposure but whose responses all sit on a boundary the family declares
        for that predictor has no finite effect.  ``"warn"`` (the default)
        emits a ``SeparationWarning`` naming the cells before any coefficient
        is fitted; ``"error"`` raises ``SeparationError``; ``"ignore"`` skips
        the scan.
        """
        return self._separation

    @property
    def coefficient_curvature(self) -> str:
        """Return the curvature the dense coefficient solve is asked to use.

        ``"observed"`` (the default) is Newton's method on the observed Hessian
        for every family; ``"fisher"`` requests Fisher scoring and needs a
        family with expected information.  Under either policy a materially
        indefinite terminal penalized observed Hessian falls back to penalized
        expected information when the family supplies it, and is refused otherwise.
        """
        return self._coefficient_curvature

    @property
    def predictors(self) -> tuple[Predictor, ...]:
        return _clone_predictor_templates(self._predictors)

    @property
    def discrete(self) -> bool:
        return self._discrete

    @property
    def n_bins(self) -> int | Mapping[str, int]:
        return self._n_bins

    def _n_bins_config(self) -> int | dict[str, int]:
        # `_owned_n_bins` returns an `int` or a `MappingProxyType`, never both,
        # so testing for the scalar and taking the mapping on the other branch
        # is the same split as testing for `Mapping` -- and it leaves a plain
        # mapping rather than one still intersected with `int`.
        n_bins = self._n_bins
        return n_bins if isinstance(n_bins, int) else dict(n_bins)

    def _fit(
        self,
        X: FrameLike | EagerFrame,
        y: NDArray,
        *,
        sample_weight: NDArray | None,
        offsets: Mapping[str, NDArray] | None,
        lambdas: Mapping[str, float] | None,
        solver_config: DenseSolverConfig,
        efs_config: DistributionalEFSConfig | None,
        retain_rows: bool,
        phase_recorder: FitPhaseRecorder | None = None,
    ) -> SuperLSS:
        next_revision = 1 if self._model is None else self._model.fit_state.revision + 1
        recorder = FitPhaseRecorder() if phase_recorder is None else phase_recorder
        before = recorder.snapshot()
        with solver_blas_threads():
            try:
                candidate = fit_dense_distributional(
                    X,
                    y,
                    family=self._family,
                    predictors=self._predictors,
                    sample_weight=sample_weight,
                    weight_contract=self._weight_contract,
                    offsets=offsets,
                    lambdas={} if lambdas is None and efs_config is None else lambdas,
                    config=solver_config,
                    efs_config=efs_config,
                    retain_rows=retain_rows,
                    discrete=self._discrete,
                    n_bins=self._n_bins_config(),
                    chunk_size="auto" if self._discrete else None,
                    phase_recorder=recorder,
                    separation=self._separation,
                    revision=next_revision,
                )
            except RepeatedCurvatureIndefinitenessError as failure:
                translated = _diagnosed_repeated_curvature_failure(
                    self._family,
                    y,
                    sample_weight=sample_weight,
                    weight_contract=self._weight_contract,
                    failure=failure,
                )
                if translated is not None:
                    raise translated from failure
                raise
        phase_snapshot = _phase_delta(recorder.snapshot(), before)
        self._model = candidate
        self._fit_phase_snapshot = phase_snapshot
        self._training_frame = X
        return self

    def fit(
        self,
        X: FrameLike | EagerFrame,
        y: NDArray,
        sample_weight: NDArray | None = None,
        offsets: Mapping[str, NDArray] | None = None,
        *,
        lambdas: Mapping[str, float] | None = None,
        max_inner_iter: int = 100,
        inner_tol: float = 1.0e-7,
        retain_rows: bool = True,
    ) -> SuperLSS:
        """Fit coefficients for caller-fixed, fully qualified smoothing parameters."""
        return self._fit(
            X,
            y,
            sample_weight=sample_weight,
            offsets=offsets,
            lambdas=lambdas,
            solver_config=DenseSolverConfig(
                max_iterations=max_inner_iter,
                tolerance=inner_tol,
                coefficient_curvature=self._coefficient_curvature,
            ),
            efs_config=None,
            retain_rows=retain_rows,
        )

    def fit_reml(
        self,
        X: FrameLike | EagerFrame,
        y: NDArray,
        sample_weight: NDArray | None = None,
        offsets: Mapping[str, NDArray] | None = None,
        *,
        method: str = "efs",
        max_reml_iter: int = 100,
        reml_tol: float = 1.0e-6,
        max_log_step: float = 5.0,
        max_lambda: float = 1.0e10,
        initial_lambda: float = 0.1,
        max_inner_iter: int = 100,
        inner_tol: float = 1.0e-7,
        lambdas: Mapping[str, float] | None = None,
        retain_rows: bool = True,
        acceleration: Literal["none", "multisecant"] = "none",
        acceleration_history: int = 5,
        acceleration_max_amplification: float = 8.0,
        practical_reml: bool = True,
        practical_reml_parameter_tol: float = 1.0e-3,
        reml_plateau_tol: float = 1.0e-7,
        outer: Literal["efs", "efs+newton"] = "efs",
        phase_recorder: FitPhaseRecorder | None = None,
    ) -> SuperLSS:
        """Fit coefficients and smoothing parameters by REML.

        ``outer`` selects the smoothing optimiser.  The default ``"efs"`` runs
        the generalised Fellner--Schall fixed point.  ``"efs+newton"`` opts into
        a Newton endgame on the exact LAML gradient and Hessian in log lambda.

        ``practical_reml`` stops the Fellner--Schall loop after sustained
        negligible objective and fitted-parameter movement. Set it to ``False``
        when strict Fellner--Schall stationarity is required.

        ``initial_lambda`` is the search start for every smoothing parameter
        without an explicit entry in ``lambdas``; the Fellner--Schall stops are
        start-dependent on some problems, so vary it when comparing fits.
        """
        if method != "efs":
            raise NotImplementedError("SuperLSS currently supports only method='efs'")
        if outer not in ("efs", "efs+newton"):
            raise ValueError("outer must be 'efs' or 'efs+newton'")
        if (
            isinstance(initial_lambda, bool)
            or not isinstance(initial_lambda, int | float)
            or not math.isfinite(initial_lambda)
            or initial_lambda <= 0.0
        ):
            raise ValueError("initial_lambda must be a finite positive number")
        return self._fit(
            X,
            y,
            sample_weight=sample_weight,
            offsets=offsets,
            lambdas=lambdas,
            solver_config=DenseSolverConfig(
                max_iterations=max_inner_iter,
                tolerance=inner_tol,
                coefficient_curvature=self._coefficient_curvature,
            ),
            efs_config=DistributionalEFSConfig(
                max_iterations=max_reml_iter,
                tolerance=reml_tol,
                max_log_step=max_log_step,
                maximum_lambda=max_lambda,
                initial_lambda=min(float(initial_lambda), max_lambda),
                acceleration=acceleration,
                acceleration_history=acceleration_history,
                acceleration_max_amplification=acceleration_max_amplification,
                practical_convergence=practical_reml,
                practical_parameter_tolerance=practical_reml_parameter_tol,
                plateau_tolerance=reml_plateau_tol,
                outer=outer,
            ),
            retain_rows=retain_rows,
            phase_recorder=phase_recorder,
        )

    def _require_fitted(self) -> DenseDistributionalModel:
        if self._model is None:
            raise RuntimeError("SuperLSS is not fitted")
        return self._model

    def diagnose(self) -> FitDiagnosticReport:
        """Explain solver and smoothing behavior from the accepted fitted revision.

        The report leads with the work profile of the fit this object ran:
        phase timing shares, iteration and refit counts, and the terminal
        state of every smoothing component.  A model restored from an
        artifact carries no machine timing, and its report says so.
        """
        return diagnose_distributional_fit(
            self._require_fitted(),
            phase_snapshot=self._fit_phase_snapshot,
        )

    @property
    def family_(self) -> DistributionalFamily:
        return self._require_fitted().family

    @property
    def predictors_(self) -> tuple[Predictor, ...]:
        return _clone_predictor_templates(self._require_fitted().fit_state.predictor_templates)

    @property
    def parameter_names_(self) -> tuple[str, ...]:
        return self._require_fitted().parameter_names

    @property
    def coef_(self) -> Mapping[str, float]:
        model = self._require_fitted()
        return MappingProxyType(
            dict(
                zip(
                    model.fitted_result.coefficient_names,
                    (float(value) for value in model.coefficients),
                    strict=True,
                )
            )
        )

    @property
    def coef_by_predictor_(self) -> Mapping[str, NDArray[np.float64]]:
        return self._require_fitted().predictor_coefficients

    @property
    def smoothing_parameters_(self) -> Mapping[str, float]:
        return self._require_fitted().smoothing_parameters

    @property
    def smoothing_convergence_reason_(self) -> EFSConvergenceReason | None:
        """Return how automatic smoothing stopped, or ``None`` for a fixed fit."""
        smoothing = self._require_fitted().smoothing
        return None if smoothing is None else smoothing.convergence_reason

    @property
    def smoothing_certified_(self) -> bool | None:
        """Return strict matched-certification status, or ``None`` for a fixed fit."""
        smoothing = self._require_fitted().smoothing
        return None if smoothing is None else smoothing.matched_certified

    @property
    def smoothing_unresolved_upper_bound_(self) -> tuple[str, ...] | None:
        """Return smoothing components with unresolved pressure at the finite cap."""
        smoothing = self._require_fitted().smoothing
        return None if smoothing is None else smoothing.unresolved_upper_bound

    @property
    def exact_face_components_(self) -> tuple[str, ...]:
        """Return smoothing components accepted at the exact infinity face."""
        return self._require_fitted().fit_state.exact_face_components

    @property
    def covariance_(self) -> NDArray[np.float64]:
        return self._require_fitted().covariance

    @property
    def result_(self) -> DistributionalFitResult:
        return self._require_fitted().fitted_result

    def predict_link(
        self,
        X: FrameLike | EagerFrame,
        *,
        offsets: Mapping[str, NDArray] | None = None,
    ) -> pd.DataFrame:
        """Return one link-scale column per family parameter."""
        values = self._require_fitted().predict_eta(X, offsets=offsets)
        return pd.DataFrame(
            values,
            columns=self.parameter_names_,
            index=_prediction_index(X, len(values)),
        )

    def predict_parameters(
        self,
        X: FrameLike | EagerFrame,
        *,
        offsets: Mapping[str, NDArray] | None = None,
    ) -> pd.DataFrame:
        """Return one natural-parameter column per family parameter."""
        values = self._require_fitted().predict_parameters(X, offsets=offsets)
        return pd.DataFrame(
            values,
            columns=self.parameter_names_,
            index=_prediction_index(X, len(values)),
        )

    def predict(
        self,
        X: FrameLike | EagerFrame,
        *,
        offsets: Mapping[str, NDArray] | None = None,
    ) -> NDArray[np.float64]:
        """Return the family-defined default prediction quantity."""
        return self._require_fitted().predict(X, offsets=offsets)

    def predict_cdf(
        self,
        X: FrameLike | EagerFrame,
        y: NDArray | float,
        *,
        offsets: Mapping[str, NDArray] | None = None,
    ) -> NDArray[np.float64]:
        """Return ``P(Y <= y)`` per row; ``y`` is a scalar or one value per row."""
        return self._require_fitted().predict_cdf(X, y, offsets=offsets)

    def predict_quantile(
        self,
        X: FrameLike | EagerFrame,
        p: NDArray | float,
        *,
        offsets: Mapping[str, NDArray] | None = None,
    ) -> NDArray[np.float64]:
        """Return the ``p``-quantile per row; ``p`` is a scalar or one value per row in (0, 1)."""
        return self._require_fitted().predict_quantile(X, p, offsets=offsets)

    def training_telemetry(self) -> SuperLSSTrainingTelemetry:
        """Return immutable audit metadata for the accepted fitted revision."""
        model = self._require_fitted()
        result = model.fitted_result
        dimensions = {
            state.name: state.coefficient_slice.stop - state.coefficient_slice.start
            for state in model.layout.predictors
        }
        n_parameters = len(result.parameter_names)
        return SuperLSSTrainingTelemetry(
            model_type="SuperLSS",
            family=type(model.family).__name__,
            parameter_names=result.parameter_names,
            curvature_policy=model.fit_state.requested_solver_config.coefficient_curvature,
            n_observations=model.null_model.n_observations,
            predictor_dimensions=dimensions,
            total_dimension=model.layout.n_coefficients,
            curvature_channels=n_parameters * (n_parameters + 1) // 2,
            discrete=self._discrete,
            n_bins=self._n_bins_config() if self._discrete else None,
            inner_iterations=result.n_inner_iter,
            smoothing_iterations=result.n_smoothing_iter,
            smoothing_convergence_reason=self.smoothing_convergence_reason_,
            smoothing_certified=self.smoothing_certified_,
            smoothing_unresolved_upper_bound=self.smoothing_unresolved_upper_bound_,
            converged=result.converged,
            rank=result.rank,
            backtracking_steps=model.result.backtracking_steps,
            curvature=result.curvature_telemetry,
        )

    def to_bytes(self) -> bytes:
        """Serialize one fitted revision; load artifacts only from trusted sources."""
        fitted = self._require_fitted()
        state = fitted.fit_state
        facade_n_bins = self._n_bins_config()
        state_n_bins = (
            state.requested_n_bins
            if isinstance(state.requested_n_bins, int)
            else dict(state.requested_n_bins)
        )
        expected_chunk_size = "auto" if self._discrete else None
        if (
            self._weight_contract != state.weight_contract
            or self._discrete is not state.requested_discrete
            or facade_n_bins != state_n_bins
            or state.requested_chunk_size != expected_chunk_size
        ):
            raise DistributionalSerializationError(
                "SuperLSS facade configuration does not match accepted fitted state"
            )
        artifact = json.loads(serialize_distributional_model(fitted))
        config = {
            "discrete": state.requested_discrete,
            "n_bins": state_n_bins,
            "separation": self._separation,
            "weight_contract": {
                "schema_version": state.weight_contract.schema_version,
                "semantics": state.weight_contract.semantics,
            },
        }
        artifact["public_api"] = {
            "artifact_type": _PUBLIC_ARTIFACT_TYPE,
            "schema_version": _PUBLIC_SCHEMA_VERSION,
            "config": config,
            "sha256": _public_config_digest(config, artifact),
        }
        return _canonical_json(artifact)

    @classmethod
    def from_bytes(
        cls,
        serialized: bytes | bytearray | memoryview | str,
    ) -> SuperLSS:
        """Restore a trusted fitted artifact after schema and integrity checks."""
        artifact, public_contract, discrete, n_bins, separation = _parse_public_artifact(serialized)
        fitted = deserialize_distributional_model(_canonical_json(artifact))
        state = fitted.fit_state
        state_n_bins = (
            state.requested_n_bins
            if isinstance(state.requested_n_bins, int)
            else dict(state.requested_n_bins)
        )
        public_n_bins = n_bins if isinstance(n_bins, int) else dict(n_bins)
        if public_contract != state.weight_contract:
            raise DistributionalSerializationError(
                "SuperLSS public weight contract does not match accepted fitted state"
            )
        if discrete is not state.requested_discrete or public_n_bins != state_n_bins:
            raise DistributionalSerializationError(
                "SuperLSS public execution configuration does not match accepted fitted state"
            )
        expected_chunk_size = "auto" if state.requested_discrete else None
        if state.requested_chunk_size != expected_chunk_size:
            raise DistributionalSerializationError(
                "SuperLSS fitted chunk policy is incompatible with its public configuration"
            )
        model = cls(
            family=fitted.family,
            predictors=fitted.fit_state.predictor_templates,
            weight_semantics=state.weight_contract.semantics,
            discrete=state.requested_discrete,
            n_bins=state_n_bins,
            separation=separation,
            coefficient_curvature=state.requested_solver_config.coefficient_curvature,
        )
        model._model = fitted
        return model

    # ------------------------------------------------------------------ #
    # Inference suite
    #
    # Every method below is a delegation to a builder in
    # ``superglm.distributional``; no statistics live in this module.  Two
    # things are resolved here and nowhere else: which frame the term grids
    # sweep, and which slot a ``sample_weight`` belongs in under the model's
    # declared weight contract.
    # ------------------------------------------------------------------ #

    def _retained_positions(
        self, n_observations: int, sample_weight: NDArray | None
    ) -> NDArray[np.intp] | None:
        """Rows a residual payload keeps, or ``None`` when it keeps every row.

        A zero weight drops a row from the fit, and the residual payload drops
        it too, so a covariate given beside ``X`` has to be cut the same way
        before it can be binned against those residuals.
        """
        if sample_weight is None:
            return None
        resolved = resolve_likelihood_weights(
            sample_weight, n_observations=n_observations, contract=self._weight_contract
        )
        positions = np.asarray(resolved.input_positions, dtype=np.intp)
        return None if len(positions) == n_observations else positions

    def _retained_design(
        self,
        X: FrameLike | EagerFrame,
        sample_weight: NDArray | None,
        offsets: Mapping[str, NDArray] | None,
    ) -> tuple[EagerFrame, Mapping[str, NDArray] | None]:
        """The frame and offsets on the rows a residual payload keeps."""
        frame = as_eager_frame(X)
        positions = self._retained_positions(len(frame), sample_weight)
        if positions is None:
            return frame, offsets
        return (
            as_eager_frame(frame.take_rows(positions)),
            _take_unvalidated_offsets(_unvalidated_offset_shapes(offsets, len(frame)), positions),
        )

    def _covariate_values(
        self,
        X: FrameLike | EagerFrame,
        covariate: str | NDArray,
        *,
        name: str | None = None,
    ) -> tuple[NDArray, str]:
        """One value per row of ``X`` from a column name or an array, and its label."""
        frame = as_eager_frame(X)
        if isinstance(covariate, str):
            if covariate not in {str(column) for column in frame.columns}:
                raise ValueError(f"covariate {covariate!r} is not a column of X")
            return frame.column_array(covariate), covariate if name is None else str(name)
        values = np.asarray(covariate)
        if values.shape != (len(frame),):
            raise ValueError("covariate must name a column of X or give one value per row of X")
        return values, "covariate" if name is None else str(name)

    def _prior_law_weight(self, sample_weight: NDArray | None) -> NDArray | None:
        """The weight a builder reads as part of the row's own law, or ``None``.

        Under ``weight_semantics="prior"`` a weight is inside the row's
        distribution -- the Gaussian variance is ``sigma^2 / w``, the Tweedie
        dispersion ``phi / w`` -- so it belongs in the builder's ``weights=``
        slot.  Under frequency semantics it is replication and says nothing
        about one row's law, so nothing is forwarded.
        """
        if sample_weight is None or self._weight_contract.semantics != "prior":
            return None
        return sample_weight

    def _refuse_frequency_row_law(self, sample_weight: NDArray | None) -> None:
        """Refuse a frequency weight where the builder has only a row-law slot."""
        if sample_weight is not None and self._weight_contract.semantics != "prior":
            raise ValueError(
                "under frequency semantics a sample_weight is replication, not part of a "
                "row's law, and this method has nowhere to put it: the predictive law of a "
                "row is its unit-weight one.  Expand the replicated rows, or declare "
                "weight_semantics='prior' if the weight is an exposure."
            )

    def _term_training_frame(
        self, X_train: FrameLike | EagerFrame | None
    ) -> FrameLike | EagerFrame:
        """The frame the term grids sweep: the one given, else the fitted one."""
        if X_train is not None:
            return X_train
        if self._training_frame is None:
            raise RuntimeError(
                "term inference sweeps a covariate over its training range, and this model "
                "carries no training frame -- a model restored with from_bytes never does. "
                "Pass X_train= the frame the model was fitted on."
            )
        return self._training_frame

    def _parameter_terms(self, parameter: str, asked: tuple[str, ...] | None) -> tuple[str, ...]:
        """Terms of one predictor that have a one-dimensional effect grid."""
        fitted = self._require_fitted()
        fitted.layout.predictor(parameter)
        prefix = f"{parameter}:"
        names = tuple(
            qualified[len(prefix) :]
            for qualified in fitted.layout.term_slices
            if qualified.startswith(prefix)
        )
        if asked is not None:
            return tuple(name for name in names if name in asked)
        return tuple(name for name in names if _compiled_spec(fitted, parameter, name) is not None)

    # -- residuals and checks ------------------------------------------- #

    def residual_set(
        self,
        X: FrameLike | EagerFrame,
        y: NDArray,
        *,
        kind: ResidualKind = "quantile",
        sample_weight: NDArray | None = None,
        offsets: Mapping[str, NDArray] | None = None,
        seed: int = 42,
    ) -> ResidualSet:
        """Return the full residual payload of ``y`` under the fitted parameters.

        The payload carries the probability-integral transform of Dunn and
        Smyth (1996) and its normal inverse together with the parameters, the
        response and the weights they were read under, which is what the Q-Q,
        worm, PIT and binned checks all consume.  ``sample_weight`` is read
        under the model's declared contract: a prior weight is part of each
        row's law, so the transform is the prior-weighted one, while a
        frequency weight is replication that the checks expand.
        """
        return compute_residuals(
            self._require_fitted(),
            X,
            y,
            kind=kind,
            sample_weight=sample_weight,
            offsets=offsets,
            seed=seed,
        )

    def residuals(
        self,
        X: FrameLike | EagerFrame,
        y: NDArray,
        *,
        kind: ResidualKind = "quantile",
        sample_weight: NDArray | None = None,
        offsets: Mapping[str, NDArray] | None = None,
        seed: int = 42,
    ) -> NDArray[np.float64]:
        """Return one residual per row: ``"quantile"`` or the raw ``"pit"`` value.

        This is :meth:`residual_set` with everything but the array discarded.
        A correct family makes the PIT values uniform and the quantile
        residuals standard normal, so these are the values every distributional
        diagnostic in the suite is drawn from.
        """
        return self.residual_set(
            X,
            y,
            kind=kind,
            sample_weight=sample_weight,
            offsets=offsets,
            seed=seed,
        ).values(kind)

    def check(
        self,
        X: FrameLike | EagerFrame,
        y: NDArray,
        covariate: str | NDArray,
        *,
        name: str | None = None,
        sample_weight: NDArray | None = None,
        offsets: Mapping[str, NDArray] | None = None,
        seed: int = 42,
        **kwargs: Any,
    ) -> BinnedCheck:
        """Return the mean, standard deviation and skewness of the residuals per bin.

        ``covariate`` is a column name of ``X`` or an array with one value per
        row; the rows are binned as Fasiolo, Nedellec, Goude and Wood (2020)
        bin them and each moment gets a bootstrap band, so a band clear of zero
        (mean), of one (standard deviation) or of zero (skewness) marks the
        region where the fit is wrong and in which moment.  ``sample_weight``
        is the aggregation weight of the residual payload; further keywords
        (``n_bins``, ``n_boot``) reach the builder.
        """
        values, label = self._covariate_values(X, covariate, name=name)
        positions = self._retained_positions(len(as_eager_frame(X)), sample_weight)
        residuals = self.residual_set(X, y, sample_weight=sample_weight, offsets=offsets, seed=seed)
        return binned_check(
            residuals,
            values if positions is None else values[positions],
            name=label,
            seed=seed,
            **kwargs,
        )

    def check_2d(
        self,
        X: FrameLike | EagerFrame,
        y: NDArray,
        covariate: str | NDArray,
        other: str | NDArray,
        *,
        names: tuple[str, str] | None = None,
        sample_weight: NDArray | None = None,
        offsets: Mapping[str, NDArray] | None = None,
        seed: int = 42,
        **kwargs: Any,
    ) -> BinnedCheck2D:
        """Return the mean residual on a two-dimensional grid of two covariates.

        Each covariate is a column name of ``X`` or an array with one value per
        row.  The tile means say where in the joint region the fit drifts, which
        a pair of one-dimensional checks cannot show; ``n_bins`` is a pair, one
        count per axis.
        """
        first, first_name = self._covariate_values(
            X, covariate, name=None if names is None else names[0]
        )
        second, second_name = self._covariate_values(
            X, other, name=None if names is None else names[1]
        )
        positions = self._retained_positions(len(as_eager_frame(X)), sample_weight)
        if positions is not None:
            first, second = first[positions], second[positions]
        residuals = self.residual_set(X, y, sample_weight=sample_weight, offsets=offsets, seed=seed)
        return binned_check_2d(residuals, first, second, names=(first_name, second_name), **kwargs)

    def actual_expected(
        self,
        X: FrameLike | EagerFrame,
        y: NDArray,
        covariate: str | NDArray,
        *,
        name: str | None = None,
        sample_weight: NDArray | None = None,
        offsets: Mapping[str, NDArray] | None = None,
        **kwargs: Any,
    ) -> ActualExpected:
        """Return the realised against the predicted total per bin of a covariate.

        Every number in the table is a ratio of weighted sums -- ``sum w y``
        over ``sum w mu_hat`` -- so on a rate target with exposure weights it
        reads as total cost over total expected cost, never as the mean of
        per-row ratios.  ``sample_weight`` is that aggregation weight, and it
        also fixes the standard error's law: prior weights put the weight
        inside each row's distribution, frequency weights replicate the row.
        """
        values, label = self._covariate_values(X, covariate, name=name)
        return actual_expected_check(
            self._require_fitted(),
            X,
            y,
            values,
            name=label,
            sample_weight=sample_weight,
            offsets=offsets,
            **kwargs,
        )

    def calibration(
        self,
        X: FrameLike | EagerFrame,
        y: NDArray,
        *,
        sample_weight: NDArray | None = None,
        offsets: Mapping[str, NDArray] | None = None,
        **kwargs: Any,
    ) -> CalibrationPayload:
        """Return interval coverage, tail totals, quantile calibration and reliability.

        Four questions in one payload, in the sense of Gneiting, Balabdaoui and
        Raftery (2007): does the central interval hold at each level, does the
        expected count of exceedances match the realised one at each threshold,
        does each predicted quantile leave the right fraction of rows above it,
        and is the exceedance forecast reliable (the CORP diagram of
        Dimitriadis, Gneiting and Jordan 2021).  ``sample_weight`` is the
        aggregation weight; ``levels``, ``thresholds`` and ``quantile_grid``
        reach the builder.
        """
        return calibration_payload(
            self._require_fitted(),
            X,
            y,
            sample_weight=sample_weight,
            offsets=offsets,
            **kwargs,
        )

    def scores(
        self,
        X: FrameLike | EagerFrame,
        y: NDArray,
        *,
        which: Sequence[str] = ("log", "crps"),
        thresholds: Sequence[float] = (),
        sample_weight: NDArray | None = None,
        offsets: Mapping[str, NDArray] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Return one column of proper scores per requested rule, one row per row.

        The log score and the continuous ranked probability score of Gneiting
        and Raftery (2007) are both proper, so a lower mean is evidence for the
        model that produced it; ``thresholds`` adds one threshold-weighted CRPS
        column per value, which scores the tail above that point alone.  Every
        column reads ``sample_weight`` under the fitted model's contract: prior
        weights select the row's weighted predictive law, while frequency
        weights return the compressed contribution of the repeated unit law.
        """
        return score_table(
            self._require_fitted(),
            X,
            y,
            which=which,
            thresholds=thresholds,
            sample_weight=sample_weight,
            offsets=offsets,
            **kwargs,
        )

    def compare(
        self,
        other: SuperLSS | DenseDistributionalModel,
        X: FrameLike | EagerFrame,
        y: NDArray,
        *,
        which: ScoreName = "log",
        by: str | Sequence[Any] | NDArray | None = None,
        sample_weight: NDArray | None = None,
        **kwargs: Any,
    ) -> Comparison:
        """Return the paired score difference against another fitted candidate.

        The comparison is paired row by row, so its standard error is that of
        the mean difference and not of two independent means; ``by`` splits it
        into segments, and ``murphy_quantile`` adds the Murphy diagram of Ehm,
        Gneiting, Jordan and Krüger (2016), which shows at which thresholds one
        candidate's advantage actually comes from.  ``sample_weight`` follows
        the candidates' declared likelihood-weight semantics; incompatible
        non-unit semantics are refused.  A negative mean difference favours
        this model.
        """
        candidate = other._require_fitted() if isinstance(other, SuperLSS) else other
        return compare_models(
            self._require_fitted(),
            candidate,
            X,
            y,
            which=which,
            by=by,
            sample_weight=sample_weight,
            **kwargs,
        )

    # -- per-parameter term inference ------------------------------------ #

    def term_inference(
        self,
        parameter: str,
        term: str,
        *,
        covariance: CovarianceKind = "fixed",
        X_train: FrameLike | EagerFrame | None = None,
        **kwargs: Any,
    ) -> ParameterTermEffect:
        """Return one term of one parameter on that parameter's link scale.

        The term is swept over its training range with every other covariate
        held at its training centre, and carries the Bayesian pointwise band of
        Marra and Wood (2012) beside the max-deviation simultaneous band of
        Ruppert, Wand and Carroll (2003).  The sweep needs the frame the model
        was fitted on: it is kept by reference at ``fit``/``fit_reml`` but does
        not survive serialization, so a restored model must be given
        ``X_train=``.
        """
        return term_effect(
            self._require_fitted(),
            self._term_training_frame(X_train),
            parameter,
            term,
            covariance=covariance,
            **kwargs,
        )

    def term_test(
        self,
        parameter: str,
        term: str,
        *,
        covariance: CovarianceKind = "fixed",
        X_train: FrameLike | EagerFrame | None = None,
        **kwargs: Any,
    ) -> TermTest:
        """Return the Wood (2013) test that one term of one parameter is flat.

        The statistic is a Wald form on the rank-truncated pseudo-inverse of
        the term's Bayesian covariance, with the rank tied to the term's
        effective degrees of freedom; because the scale is itself modelled the
        reference is a chi-squared and not an F.  Like :meth:`term_inference`
        it evaluates on the training frame, so a restored model needs
        ``X_train=``.
        """
        return term_test(
            self._require_fitted(),
            self._term_training_frame(X_train),
            parameter,
            term,
            covariance=covariance,
            **kwargs,
        )

    def summary(
        self,
        *,
        covariance: CovarianceKind = "fixed",
        X_train: FrameLike | EagerFrame | None = None,
    ) -> pd.DataFrame:
        """Return one row per intercept and per term of every parameter.

        The columns are the term's effective degrees of freedom, its smoothing
        parameter where it has exactly one, the Wood (2013) statistic with its
        rank and p-value, and the estimate and standard error where the term
        holds a single coefficient.  It reads the training frame the same way
        :meth:`term_test` does, so a restored model needs ``X_train=``.
        """
        return summary_table(
            self._require_fitted(),
            self._term_training_frame(X_train),
            covariance=covariance,
        )

    # -- the posterior primitive ----------------------------------------- #

    def posterior_draws(
        self,
        n_draws: int = 1000,
        *,
        covariance: CovarianceKind = "fixed",
        seed: int = 42,
    ) -> PosteriorDraws:
        """Return coefficient draws from the Bayesian posterior of the fit.

        The draws are ``N(beta_hat, V)`` in the fit's own global coordinates,
        which is the posterior of Marra and Wood (2012) with the smoothing
        parameters held at their estimates; ``covariance="corrected"`` asks
        instead for the smoothing-uncertainty correction of Wood, Pya and
        Säfken (2016), using either trusted published curvature or one
        authenticated replay from a stationary fit's retained rows.  Compact
        fits without published curvature and untrusted fits refuse.  One draw
        set can be reused across quantities.
        """
        return posterior_draws(self._require_fitted(), n_draws, covariance=covariance, seed=seed)

    def posterior_bounds(
        self,
        X: FrameLike | EagerFrame,
        quantity: Quantity,
        *,
        sample_weight: NDArray | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame | tuple[pd.DataFrame, NDArray[np.float64]]:
        """Return per-row bounds on a quantity derived from the fitted parameters.

        ``quantity`` names what to push the coefficient draws through -- a
        parameter, a predictive quantile, an exceedance probability, an
        expected shortfall, or a callable of the parameter matrix -- and the
        frame reports the plug-in estimate beside the posterior mean, standard
        deviation and interval.  ``sample_weight`` is forwarded as the row's
        prior weight, since a quantile of a row at a fifth of a year's exposure
        is the quantile of its own law; under frequency semantics there is no
        such slot and the call is refused.
        """
        self._refuse_frequency_row_law(sample_weight)
        return posterior_bounds(
            self._require_fitted(),
            X,
            quantity,
            weights=self._prior_law_weight(sample_weight),
            **kwargs,
        )

    def posterior_predictive(
        self,
        X: FrameLike | EagerFrame,
        n_draws: int = 200,
        *,
        sample_weight: NDArray | None = None,
        **kwargs: Any,
    ) -> NDArray[np.float64]:
        """Simulate responses for the rows of ``X`` through the family's quantile.

        With ``parameter_uncertainty=True`` each draw uses its own coefficient
        draw, so the spread is predictive and not merely conditional; ``reduce``
        collapses each block of draws before it is materialised, which is what
        makes a book-level total affordable.  ``sample_weight`` enters as the
        row's prior weight, and under frequency semantics the call is refused
        rather than simulating a law the weight does not describe.
        """
        self._refuse_frequency_row_law(sample_weight)
        return posterior_predictive(
            self._require_fitted(),
            X,
            n_draws,
            weights=self._prior_law_weight(sample_weight),
            **kwargs,
        )

    # -- surfaces, spread and the book ----------------------------------- #

    def risk_curves(
        self,
        reference: Mapping[str, Any] | pd.Series,
        covariate: str,
        *,
        X_train: FrameLike | EagerFrame | None = None,
        **kwargs: Any,
    ) -> RiskCurves:
        """Return predicted response quantiles along one covariate, with bands.

        One reference row is swept over the covariate's training range while
        every other column is held at ``reference`` (or at its training centre),
        and each requested quantile of the predictive law is reported with a
        posterior band drawn from one shared draw set, so the curves are
        coherent with one another.  ``weights`` here are per swept point, not
        per training row: they state the exposure the curve is priced at.
        """
        return risk_curves(
            self._require_fitted(),
            self._term_training_frame(X_train),
            reference,
            covariate,
            **kwargs,
        )

    def density_fan(
        self,
        reference: Mapping[str, Any] | pd.Series,
        covariate: str,
        *,
        X_train: FrameLike | EagerFrame | None = None,
        **kwargs: Any,
    ) -> DensityFan:
        """Return the conditional response density along one covariate.

        The same sweep as :meth:`risk_curves`, but the whole density at each
        point rather than a few quantiles of it: it is the picture that shows a
        shape change -- a mass moving into the tail, a mode splitting -- which
        no set of quantile curves states outright.
        """
        return density_fan(
            self._require_fitted(),
            self._term_training_frame(X_train),
            reference,
            covariate,
            **kwargs,
        )

    def parameter_spread(
        self,
        X: FrameLike | EagerFrame,
        *,
        threshold: float,
        sample_weight: NDArray | None = None,
        **kwargs: Any,
    ) -> Spread:
        """Return how far the fitted parameters spread, and how far identical prices do.

        The histograms show the sharpness of each fitted parameter and of the
        predicted tail quantile over the rows; the identically-priced table
        bins rows by predicted mean and reports the spread of
        ``P(Y > threshold)`` inside each bin, which is the quantity a
        location-only model has no way to distinguish.  Under prior semantics
        ``sample_weight`` enters twice, because it means the same thing in both
        places: it weighs the ratio of sums the table reports and it is part of
        each row's own law.
        """
        return parameter_spread(
            self._require_fitted(),
            X,
            threshold=threshold,
            sample_weight=sample_weight,
            weights=self._prior_law_weight(sample_weight),
            **kwargs,
        )

    def portfolio(
        self,
        X: FrameLike | EagerFrame,
        *,
        sample_weight: NDArray | None = None,
        **kwargs: Any,
    ) -> Portfolio:
        """Return the simulated total over a book of rows, optionally by segment.

        Each row is simulated on its own predictive law and the draws are summed
        across rows, so the reported quantiles are of the book total and carry
        the dependence the shared coefficient draws induce.  ``sample_weight``
        enters twice under prior semantics -- every row is simulated on its own
        prior-weighted law and what the book pays is ``sum w y`` -- and is
        refused under frequency semantics, which has no row-law slot.
        """
        self._refuse_frequency_row_law(sample_weight)
        return portfolio(
            self._require_fitted(),
            X,
            weights=self._prior_law_weight(sample_weight),
            **kwargs,
        )

    # -- figures and their payloads -------------------------------------- #

    def plot_diagnostics(
        self,
        X: FrameLike | EagerFrame,
        y: NDArray,
        *,
        engine: str = "matplotlib",
        n_sim: int = 100,
        max_points: int = 50_000,
        seed: int = 42,
        sample_weight: NDArray | None = None,
        offsets: Mapping[str, NDArray] | None = None,
    ) -> Any:
        """Draw the six-panel distributional diagnostic.

        Q-Q with a simulated envelope, the worm plot, the PIT histogram, the
        residual density against the standard normal, the residuals against the
        first parameter's linear predictor and their standard deviation in bins
        of the second's: the first three ask whether the family is right, the
        last three where it is wrong.  ``sample_weight`` is the aggregation
        weight of the residuals underneath; ``engine`` selects matplotlib or
        plotly, and only the engine asked for is imported.
        """
        render = _renderer(engine, "diagnostics")
        fitted = self._require_fitted()
        residuals = self.residual_set(X, y, sample_weight=sample_weight, offsets=offsets, seed=seed)
        frame, resolved_offsets = self._retained_design(X, sample_weight, offsets)
        qq = qq_payload(
            fitted,
            residuals,
            n_sim=n_sim,
            max_points=max_points,
            seed=seed,
            X=frame,
            offsets=resolved_offsets,
        )
        return render(
            qq,
            worm_payload(residuals),
            pit_payload(residuals),
            residuals,
            max_points=max_points,
        )

    def plot(
        self,
        parameter: str | None = None,
        terms: str | Sequence[str] | None = None,
        *,
        engine: str = "matplotlib",
        covariance: CovarianceKind = "fixed",
        X_train: FrameLike | EagerFrame | None = None,
        **kwargs: Any,
    ) -> Any:
        """Draw a grid of term panels per parameter, pointwise and simultaneous bands.

        With ``parameter`` named the return is that parameter's figure; with
        ``parameter=None`` every parameter that has a plottable term gets one
        and they come back as a dict keyed by parameter name, which is the view
        that shows what drives the location against what drives the scale.
        ``terms`` restricts the grid; interaction terms have no one-dimensional
        grid and are skipped unless named, in which case the builder refuses.
        """
        render = _renderer(engine, "term_grid")
        fitted = self._require_fitted()
        frame = self._term_training_frame(X_train)
        names = (
            tuple(state.name for state in fitted.layout.predictors)
            if parameter is None
            else (parameter,)
        )
        asked = None if terms is None else (terms,) if isinstance(terms, str) else tuple(terms)
        selections: dict[str, tuple[str, ...]] = {}
        for name in names:
            selected = self._parameter_terms(name, asked)
            if not selected:
                if parameter is not None:
                    asked_for = "" if asked is None else f", among the terms asked for: {asked}"
                    raise ValueError(
                        f"parameter {name!r} has no term with a one-dimensional effect grid"
                        f"{asked_for}"
                    )
                continue
            selections[name] = selected
        if not selections:
            raise ValueError("no parameter has a term with a one-dimensional effect grid")

        prepared = {
            name: tuple(
                _prepare_term_effect(
                    fitted,
                    frame,
                    name,
                    term,
                    covariance=covariance,
                    **kwargs,
                )
                for term in selected
            )
            for name, selected in selections.items()
        }
        matrix = posterior_covariance(fitted, kind=covariance)
        figures: dict[str, Any] = {}
        for name, panels in prepared.items():
            figures[name] = render(
                [
                    _term_effect_from_covariance(
                        fitted,
                        panel,
                        matrix,
                    )
                    for panel in panels
                ],
                parameter=name,
            )
        if parameter is not None:
            return figures[parameter]
        return figures

    def plot_data(self, kind: str, **kwargs: Any) -> Any:
        """Return the JSON-clean payload behind one figure, without drawing it.

        ``kind`` names the payload -- ``"qq"``, ``"worm"``, ``"pit"``,
        ``"binned"``, ``"actual_expected"``, ``"calibration"``, ``"scores"``,
        ``"comparison"``, ``"term"``, ``"risk_curves"``, ``"density_fan"``,
        ``"spread"`` or ``"portfolio"`` -- and the keywords are those of the
        method that builds it.  Every payload carries what was asked of it
        (levels, seeds, bin edges, draw counts), so a figure is reproducible
        from its payload alone and a front end can draw it without the model.
        """
        if kind not in _PLOT_DATA_KINDS:
            raise ValueError(f"kind must be one of {_PLOT_DATA_KINDS}, not {kind!r}")
        if kind == "scores":
            return [
                {name: _json_number(value) for name, value in row.items()}
                for row in self.scores(**kwargs).to_dict(orient="records")
            ]
        builders: Mapping[str, Any] = {
            "qq": self._qq_payload,
            "worm": self._worm_payload,
            "pit": self._pit_payload,
            "binned": self.check,
            "actual_expected": self.actual_expected,
            "calibration": self.calibration,
            "comparison": self.compare,
            "term": self.term_inference,
            "risk_curves": self.risk_curves,
            "density_fan": self.density_fan,
            "spread": self.parameter_spread,
            "portfolio": self.portfolio,
        }
        return builders[kind](**kwargs).to_json()

    def _qq_payload(
        self,
        X: FrameLike | EagerFrame,
        y: NDArray,
        *,
        sample_weight: NDArray | None = None,
        offsets: Mapping[str, NDArray] | None = None,
        seed: int = 42,
        **kwargs: Any,
    ) -> QQPayload:
        """The Q-Q payload with its simulated envelope, on the retained rows."""
        residuals = self.residual_set(X, y, sample_weight=sample_weight, offsets=offsets, seed=seed)
        frame, resolved_offsets = self._retained_design(X, sample_weight, offsets)
        return qq_payload(
            self._require_fitted(),
            residuals,
            seed=seed,
            X=frame,
            offsets=resolved_offsets,
            **kwargs,
        )

    def _worm_payload(
        self,
        X: FrameLike | EagerFrame,
        y: NDArray,
        *,
        covariate: str | NDArray | None = None,
        covariate_name: str | None = None,
        sample_weight: NDArray | None = None,
        offsets: Mapping[str, NDArray] | None = None,
        seed: int = 42,
        **kwargs: Any,
    ) -> WormPayload:
        """The worm payload, one panel per interval of ``covariate`` when given."""
        residuals = self.residual_set(X, y, sample_weight=sample_weight, offsets=offsets, seed=seed)
        values: NDArray | None = None
        label = covariate_name
        if covariate is not None:
            values, label = self._covariate_values(X, covariate, name=covariate_name)
            positions = self._retained_positions(len(as_eager_frame(X)), sample_weight)
            if positions is not None:
                values = values[positions]
        return worm_payload(residuals, covariate=values, covariate_name=label, **kwargs)

    def _pit_payload(
        self,
        X: FrameLike | EagerFrame,
        y: NDArray,
        *,
        sample_weight: NDArray | None = None,
        offsets: Mapping[str, NDArray] | None = None,
        seed: int = 42,
        **kwargs: Any,
    ) -> PITPayload:
        """The PIT histogram payload with its binomial band."""
        return pit_payload(
            self.residual_set(X, y, sample_weight=sample_weight, offsets=offsets, seed=seed),
            **kwargs,
        )


__all__ = ["SuperLSS", "SuperLSSTrainingTelemetry"]
