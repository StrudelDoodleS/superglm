"""Posterior simulation for a fitted distributional (location-scale-shape) model.

One primitive sits under every band, bound, envelope and predictive draw of the
inference suite: draws from the Bayesian posterior of the coefficients, their
pushforward through the fitted predictors to natural parameters, and the
derived quantities the family supplies.

The posterior is the Bayesian one at the fitted smoothing parameters,
``N(beta_hat, V)`` with ``V`` the penalised covariance, whose across-the-function
coverage is the subject of Marra and Wood (2012), *Scandinavian Journal of
Statistics* 39(1), 53-74, and Wood (2017), *Generalized Additive Models: An
Introduction with R*, 2nd ed., section 6.10.  ``kind="corrected"`` adds the
first-order smoothing-parameter term of Wood, Pya and Saefken (2016), *Journal
of the American Statistical Association* 111(516), 1548-1563, which needs a
certified terminal smoothing Hessian, either published by the Newton endgame or
replayed on demand from retained fit rows.  The max-deviation critical value of
:func:`simultaneous_critical_value` is Ruppert, Wand and Carroll (2003),
*Semiparametric Regression*, section 6.5.  Expected shortfall is the quantile
integral above the level, the representation of Acerbi and Tasche (2002),
*Journal of Banking and Finance* 26(7), 1487-1503.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from superglm._frame import EagerFrame, FrameLike, as_eager_frame
from superglm.distributional.family import (
    COMPLETE_OBSERVATION,
    DefaultPredictionFamily,
    DistributionFunctionFamily,
    ExpectedShortfallFamily,
    PriorWeightedDistributionFunctionFamily,
    PriorWeightedExpectedShortfallFamily,
)

# ``_prediction_offsets`` is the one validator of predictor-keyed offsets the
# fit and every prediction path already share; restating its rules here is how
# an offset contract drifts between two entry points that must agree.
from superglm.distributional.model import _prediction_offsets
from superglm.distributional.prediction_design import build_joint_prediction_design
from superglm.distributional.smoothing.derivatives import LamlDerivatives, laml_derivatives
from superglm.distributional.solver.assembly import dense_predictor_matrices
from superglm.reml.penalty_algebra import penalty_component_dense_matrix

CovarianceKind = Literal["fixed", "corrected"]
#: A quantity is a name, a name with one argument, or a callable on ``theta``.
Quantity = str | tuple[Any, ...] | Callable[[NDArray], NDArray]

#: Natural-parameter scratch per row chunk stays under this many bytes.
_CHUNK_BYTES = 256_000_000
#: Predictive uniforms are drawn strictly inside ``(0, 1)`` by this margin.
_PROBABILITY_MARGIN = 1.0e-12
#: Floor on a pointwise standard error before it divides a deviation.
_MINIMUM_STANDARD_ERROR = 1.0e-20
#: Refusal when a row law is asked of a family that cannot state one.
_PRIOR_WEIGHT_REFUSAL = (
    "{family} has no prior-weighted distribution function, so the row law under non-unit "
    "prior weights is unavailable; pass unit weights, declare frequency semantics, or "
    "implement PriorWeightedDistributionFunctionFamily"
)
_PRIOR_WEIGHT_SHORTFALL_REFUSAL = (
    "{family} has no prior-weighted expected shortfall, so the row law under non-unit "
    "prior weights is unavailable; pass unit weights, declare frequency semantics, or "
    "implement PriorWeightedExpectedShortfallFamily"
)


def _prior_weight_law(
    weights: NDArray | None, n_rows: int | None = None
) -> NDArray[np.float64] | None:
    """Return the weights that change the row law, or ``None`` for the unit law.

    A prior weight of one leaves the law alone, so an all-unit vector collapses
    to ``None`` and every caller then follows the unweighted path bit for bit.
    """
    if weights is None:
        return None
    values = np.asarray(weights, dtype=np.float64)
    shaped = values.ndim == 1 and (n_rows is None or values.shape == (n_rows,))
    if not shaped or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("weights must give one positive prior weight per row of X")
    return None if bool(np.all(values == 1.0)) else values


def _row_index(X: FrameLike | EagerFrame, n_observations: int) -> pd.Index:
    if isinstance(X, pd.DataFrame):
        return X.index.copy()
    if isinstance(X, EagerFrame) and isinstance(X.native, pd.DataFrame):
        return X.native.index.copy()
    return pd.RangeIndex(n_observations)


@dataclass(frozen=True)
class PosteriorDraws:
    """Coefficient draws in the qualified global coordinates of the fit."""

    coefficients: NDArray[np.float64]
    covariance_kind: CovarianceKind
    seed: int
    coefficient_names: tuple[str, ...]

    def __post_init__(self) -> None:
        values = np.array(self.coefficients, dtype=np.float64, copy=True)
        if values.ndim != 2 or values.shape[0] < 1:
            raise ValueError("posterior draws must be a (draws, coefficients) matrix")
        if not np.all(np.isfinite(values)):
            raise ValueError("posterior draws must be finite")
        names = tuple(self.coefficient_names)
        if len(names) != values.shape[1]:
            raise ValueError("coefficient_names must name every drawn coefficient")
        if self.covariance_kind not in ("fixed", "corrected"):
            raise ValueError("covariance_kind must be 'fixed' or 'corrected'")
        values.setflags(write=False)
        object.__setattr__(self, "coefficients", values)
        object.__setattr__(self, "coefficient_names", names)
        object.__setattr__(self, "seed", int(self.seed))

    @property
    def n_draws(self) -> int:
        return int(self.coefficients.shape[0])


def _smoothing_gradient_names(smoothing: Any) -> tuple[str, ...]:
    """Return the authenticated smoothing row order from terminal evidence.

    The endgame publishes the Hessian over the names of its ``LamlDerivatives``
    -- the *estimated* smoothing components in penalty order -- and uses the
    same names as the keys of ``terminal_gradient``.  Those names are an
    order-preserving subsequence of ``lambdas`` rather than all of it: a
    component with a fixed lambda policy, or one the search sent to a working
    infinity, carries a lambda but contributes no derivative row.  Both
    invariants are asserted here rather than assumed.
    """
    gradient = smoothing.terminal_gradient
    if gradient is None:
        raise RuntimeError(
            "the published smoothing Hessian carries no row order; its terminal gradient is absent"
        )
    names = tuple(gradient)
    selected = set(names)
    if tuple(name for name in smoothing.lambdas if name in selected) != names:
        raise RuntimeError("the smoothing Hessian rows do not follow the smoothing lambda order")
    return names


def _smoothing_hessian_names(smoothing: Any) -> tuple[str, ...]:
    """Return the row order of an already-published smoothing Hessian."""
    names = _smoothing_gradient_names(smoothing)
    hessian = np.asarray(smoothing.smoothing_hessian, dtype=np.float64)
    if hessian.shape != (len(names), len(names)):
        raise RuntimeError("the smoothing Hessian is not square over its published names")
    return names


def _authenticated_terminal_fit(fitted: Any, smoothing: Any) -> tuple[Any, Any]:
    """Return the immutable fit state and terminal result eligible for correction."""
    reason = smoothing.convergence_reason
    if not smoothing.converged or reason != "stationary":
        raise RuntimeError(
            "corrected covariance requires a converged stationary smoothing result; "
            f"this fit stopped with reason {reason!r} (see model.diagnose())"
        )
    try:
        state = fitted.fit_state
        terminal = fitted.result
    except AttributeError as exc:
        raise RuntimeError(
            "corrected covariance cannot authenticate the fitted terminal state"
        ) from exc
    if (
        state.smoothing is not smoothing
        or state.solver_result is not terminal
        or smoothing.terminal_fit is not terminal
    ):
        raise RuntimeError(
            "corrected covariance cannot authenticate the accepted terminal-result provenance"
        )
    state_identifier = state.family_likelihood_plan_identifier
    terminal_identifier = terminal.family_likelihood_plan_identifier
    if state_identifier != terminal_identifier:
        raise RuntimeError(
            "corrected covariance found a likelihood plan identifier mismatch between "
            "fit state and terminal result"
        )
    if dict(state.lambdas) != dict(smoothing.lambdas):
        raise RuntimeError("corrected covariance found unstable terminal smoothing provenance")
    face = terminal.coefficient_face
    face_names = () if face is None else tuple(face.component_names)
    if tuple(state.exact_face_components) != face_names:
        raise RuntimeError("corrected covariance found unstable terminal active-face provenance")
    curvature = terminal.terminal_curvature
    if (
        terminal.config.coefficient_curvature != "observed"
        or curvature.requested_source != "observed"
        or curvature.actual_source != "observed"
        or curvature.fallback_count != 0
    ):
        raise RuntimeError(
            "corrected covariance requires observed terminal curvature without fallback"
        )
    return state, terminal


def _trusted_smoothing_hessian(
    hessian: NDArray | None,
    certificate: NDArray | None,
    *,
    count: int,
    certificate_fraction: float,
    replayed: bool,
) -> NDArray[np.float64]:
    """Apply the Newton endgame's positive-definite certificate trust gate."""
    if hessian is None or certificate is None:
        if replayed:
            raise RuntimeError(
                "smoothing derivative replay returned no smoothing Hessian certificate"
            )
        raise RuntimeError("the published smoothing Hessian has no numerical certificate")
    try:
        matrix = np.asarray(hessian, dtype=np.float64)
        bound = np.asarray(certificate, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("the smoothing Hessian and its certificate must be numeric") from exc
    shape = (count, count)
    if matrix.shape != shape or bound.shape != shape:
        raise RuntimeError("the smoothing Hessian and certificate do not cover its active names")
    if (
        not np.all(np.isfinite(matrix))
        or not np.array_equal(matrix, matrix.T)
        or not np.all(np.isfinite(bound))
        or np.any(bound < 0.0)
    ):
        raise RuntimeError("the smoothing Hessian and its certificate are not finite and valid")
    try:
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(matrix)))
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            "the smoothing Hessian positive-definite check could not be resolved"
        ) from exc
    if minimum_eigenvalue <= 0.0:
        raise RuntimeError("the smoothing Hessian is not positive definite")
    trust_bar = float(certificate_fraction) * float(np.min(np.diag(matrix)))
    maximum_certificate = float(np.max(bound))
    if not trust_bar > maximum_certificate:
        raise RuntimeError(
            "the smoothing Hessian certificate is too large for corrected covariance"
        )
    return matrix


def _replayed_smoothing_hessian(
    fitted: Any,
    smoothing: Any,
    names: tuple[str, ...],
    state: Any,
    terminal: Any,
) -> NDArray[np.float64]:
    """Replay terminal smoothing curvature once, without publishing or caching it."""
    rows = state.retained_rows
    if rows is None:
        raise RuntimeError(
            "corrected covariance needs retained training rows when no smoothing Hessian "
            "was published; refit with retain_rows=True"
        )
    weights = rows.likelihood_weights
    provenance = weights.provenance
    if (
        provenance != state.weight_provenance
        or provenance.contract != state.weight_contract
        or weights.digest != provenance.root_digest
        or not np.array_equal(
            weights.root_take_map,
            np.arange(len(rows.response), dtype=np.intp),
        )
    ):
        raise RuntimeError(
            "retained training rows do not preserve the fitted root weight provenance"
        )
    try:
        likelihood_plan = fitted.family.bind_likelihood(
            rows.response,
            weights,
            COMPLETE_OBSERVATION,
        )
        replay_identifier = likelihood_plan.plan_identifier
    except Exception as exc:
        raise RuntimeError(
            "corrected covariance could not rebind the fitted likelihood plan"
        ) from exc
    if (
        replay_identifier != state.family_likelihood_plan_identifier
        or replay_identifier != terminal.family_likelihood_plan_identifier
    ):
        raise RuntimeError(
            "corrected covariance replay produced a different likelihood plan identifier"
        )
    try:
        matrices = dense_predictor_matrices(fitted.layout)
        derivatives = laml_derivatives(
            fitted.family,
            fitted.layout,
            rows.response,
            likelihood_plan,
            lambdas=state.lambdas,
            fit=terminal,
            dense_matrices=matrices,
            step=smoothing.config.derivative_step,
            want_hessian=True,
        )
    except Exception as exc:
        raise RuntimeError("smoothing derivative replay failed for corrected covariance") from exc
    if not isinstance(derivatives, LamlDerivatives):
        raise RuntimeError("smoothing derivative replay returned an invalid result")
    if derivatives.names != names:
        raise RuntimeError(
            "smoothing derivative replay did not preserve the ordered smoothing names"
        )
    face = terminal.coefficient_face
    expected_provenance = (
        terminal.terminal_rank.method,
        int(terminal.terminal_rank.rank),
        "observed",
        () if face is None else tuple(face.component_names),
    )
    if derivatives.provenance != expected_provenance:
        raise RuntimeError("smoothing derivative replay changed the terminal derivative provenance")
    stored_gradient = smoothing.terminal_gradient
    stored_certificate = smoothing.terminal_gradient_certificate
    if stored_gradient is None or stored_certificate is None:
        raise RuntimeError("stationary smoothing evidence has no terminal gradient certificate")
    if tuple(stored_gradient) != names or tuple(stored_certificate) != names:
        raise RuntimeError("stationary terminal gradient evidence has inconsistent smoothing names")
    gradient = np.array([stored_gradient[name] for name in names], dtype=np.float64)
    gradient_certificate = np.array([stored_certificate[name] for name in names], dtype=np.float64)
    if (
        not np.all(np.isfinite(gradient))
        or not np.all(np.isfinite(gradient_certificate))
        or np.any(gradient_certificate < 0.0)
    ):
        raise RuntimeError("stationary terminal gradient evidence is not numerically certified")
    disagreement = np.abs(derivatives.gradient - gradient)
    agreement_bound = derivatives.gradient_certificate + gradient_certificate
    if np.any(disagreement > agreement_bound):
        raise RuntimeError(
            "replayed smoothing derivatives disagree with the stored terminal gradient"
        )
    return _trusted_smoothing_hessian(
        derivatives.hessian,
        derivatives.hessian_certificate,
        count=len(names),
        certificate_fraction=smoothing.config.hessian_certificate_fraction,
        replayed=True,
    )


def _resolved_smoothing_hessian(
    fitted: Any,
    smoothing: Any,
    names: tuple[str, ...],
    *,
    replay_authority: tuple[Any, Any] | None,
) -> NDArray[np.float64]:
    """Use trusted published curvature or one authenticated ephemeral replay."""
    if smoothing.smoothing_hessian is None:
        if replay_authority is None:
            raise RuntimeError("corrected covariance has no authenticated terminal replay state")
        state, terminal = replay_authority
        return _replayed_smoothing_hessian(fitted, smoothing, names, state, terminal)
    return _trusted_smoothing_hessian(
        smoothing.smoothing_hessian,
        smoothing.smoothing_hessian_certificate,
        count=len(names),
        certificate_fraction=smoothing.config.hessian_certificate_fraction,
        replayed=False,
    )


def posterior_covariance(fitted: Any, *, kind: CovarianceKind = "fixed") -> NDArray[np.float64]:
    """Return the posterior coefficient covariance in global coordinates.

    ``kind="fixed"`` is the Bayesian covariance at the fitted smoothing
    parameters.  ``kind="corrected"`` adds the first-order smoothing-parameter
    term of Wood, Pya and Saefken (2016), ``V + J V_rho J'``, with
    ``J = d beta_hat / d rho`` from the implicit function theorem applied to the
    penalised score, and ``V_rho`` the inverse negative-LAML Hessian in
    ``rho = log lambda``.  It refuses rather than silently returning the fixed
    covariance when terminal smoothing curvature cannot be authenticated.  A
    stationary fit with retained training rows replays that curvature only for
    this request when the Newton endgame stopped before forming a Hessian.
    """
    covariance = np.array(fitted.inference.covariance, dtype=np.float64, copy=True)
    if kind == "fixed":
        return covariance
    if kind != "corrected":
        raise ValueError("covariance must be 'fixed' or 'corrected'")

    smoothing = fitted.smoothing
    if smoothing is None:
        raise RuntimeError(
            "corrected covariance requires a converged stationary smoothing result; "
            "this fit has no smoothing result"
        )
    terminal_authority = _authenticated_terminal_fit(fitted, smoothing)
    if smoothing.smoothing_hessian is None:
        names = _smoothing_gradient_names(smoothing)
    else:
        names = _smoothing_hessian_names(smoothing)
    components = {component.name: component for component in fitted.layout.penalties}
    missing = [name for name in names if name not in components]
    if missing:
        raise RuntimeError(f"the smoothing Hessian names no layout penalty: {missing}")
    rho_hessian = _resolved_smoothing_hessian(
        fitted,
        smoothing,
        names,
        replay_authority=(terminal_authority if smoothing.smoothing_hessian is None else None),
    )

    beta = np.asarray(fitted.result.coefficients, dtype=np.float64)
    columns = []
    for name in names:
        component = components[name]
        omega = np.asarray(penalty_component_dense_matrix(component), dtype=np.float64)
        rhs = np.zeros_like(beta)
        block = component.group_sl
        rhs[block] = float(smoothing.lambdas[name]) * (omega @ beta[block])
        columns.append(-np.asarray(fitted.result.solve_terminal(rhs), dtype=np.float64))
    jacobian = np.column_stack(columns)

    rho_covariance = np.linalg.pinv(rho_hessian, hermitian=True)
    corrected = covariance + jacobian @ rho_covariance @ jacobian.T
    return 0.5 * (corrected + corrected.T)


def _posterior_draw_count(n_draws: int) -> int:
    count = int(n_draws)
    if count < 2:
        raise ValueError("n_draws must be at least 2 to summarise a posterior")
    return count


def _posterior_draws_from_covariance(
    fitted: Any,
    matrix: NDArray,
    count: int,
    *,
    covariance: CovarianceKind = "fixed",
    seed: int = 42,
) -> PosteriorDraws:
    """Draw coefficients from one already-resolved posterior covariance."""
    resolved = np.asarray(matrix, dtype=np.float64)
    beta = np.asarray(fitted.result.coefficients, dtype=np.float64)
    if resolved.shape != (len(beta), len(beta)):
        raise ValueError("the posterior covariance does not match the fitted coefficients")

    eigenvalues, vectors = np.linalg.eigh(0.5 * (resolved + resolved.T))
    tolerance = float(fitted.inference.reconciliation_tolerance) * max(
        float(np.max(eigenvalues, initial=0.0)), 0.0
    )
    if np.any(eigenvalues < -tolerance):
        raise ValueError("the posterior covariance has a materially negative eigenvalue")
    factor = vectors * np.sqrt(np.where(eigenvalues < tolerance, 0.0, eigenvalues))

    rng = np.random.default_rng(seed)
    normals = rng.standard_normal((count, len(beta)))
    return PosteriorDraws(
        coefficients=beta + normals @ factor.T,
        covariance_kind=covariance,
        seed=int(seed),
        coefficient_names=tuple(fitted.layout.coefficient_names),
    )


def posterior_draws(
    fitted: Any,
    n_draws: int = 1000,
    *,
    covariance: CovarianceKind = "fixed",
    seed: int = 42,
) -> PosteriorDraws:
    """Draw coefficients from ``N(beta_hat, V)`` with a symmetric eigen square root.

    ``V`` is a pseudo-inverse and may be rank deficient: eigenvalues below
    ``reconciliation_tolerance * lambda_max`` are set to zero, while a
    materially negative one is an error rather than a clip.  Every draw is made
    from one ``default_rng(seed)`` stream, so the result never depends on how a
    later pushforward chunks its rows.
    """
    count = _posterior_draw_count(n_draws)
    matrix = posterior_covariance(fitted, kind=covariance)
    return _posterior_draws_from_covariance(
        fitted,
        matrix,
        count,
        covariance=covariance,
        seed=seed,
    )


def _resolved_chunk_rows(chunk_rows: int | None, *, n_draws: int, n_parameters: int) -> int:
    if chunk_rows is None:
        return max(1, int(_CHUNK_BYTES // (8 * max(n_draws, 1) * max(n_parameters, 1))))
    rows = int(chunk_rows)
    if rows < 1:
        raise ValueError("chunk_rows must be a positive number of rows")
    return rows


def posterior_parameters(
    fitted: Any,
    X: FrameLike | EagerFrame,
    draws: PosteriorDraws,
    *,
    offsets: Mapping[str, NDArray] | None = None,
    chunk_rows: int | None = None,
) -> Generator[tuple[slice, NDArray[np.float64]], None, None]:
    """Yield ``(row_slice, theta)`` blocks of shape ``(draws, rows, parameters)``.

    Each predictor's local prediction design multiplies the draw matrix on the
    link scale, the predictor offset is added there, and the fitted inverse link
    maps to the natural parameter.  The design is built once for all rows and
    sliced, so chunked and unchunked pushforwards are bit-identical.
    """
    if not isinstance(draws, PosteriorDraws):
        raise TypeError("draws must be a PosteriorDraws")
    layout = fitted.layout
    if draws.coefficient_names != tuple(layout.coefficient_names):
        raise ValueError("draw coefficient names do not match the fitted layout")

    frame = as_eager_frame(X)
    n_observations = len(frame)
    n_parameters = len(layout.predictors)
    rows_per_chunk = _resolved_chunk_rows(
        chunk_rows, n_draws=draws.n_draws, n_parameters=n_parameters
    )
    resolved_offsets = _prediction_offsets(
        offsets, tuple(state.name for state in layout.predictors), n_observations
    )
    design = build_joint_prediction_design(frame, fitted.compiled_predictors, layout)
    coefficients = draws.coefficients

    for start in range(0, n_observations, rows_per_chunk):
        stop = min(start + rows_per_chunk, n_observations)
        width = stop - start
        block = np.empty((draws.n_draws, width, n_parameters), dtype=np.float64)
        for index, state in enumerate(layout.predictors):
            local = design.local[state.name][start:stop]
            eta = local @ coefficients[:, state.coefficient_slice].T
            eta += resolved_offsets[state.name][start:stop, None]
            values = np.asarray(state.link.inverse(eta.T.reshape(-1)), dtype=np.float64)
            if values.shape != (eta.size,):
                raise ValueError(f"inverse link for {state.name!r} returned an invalid shape")
            block[:, :, index] = values.reshape(draws.n_draws, width)
        yield slice(start, stop), block


def _scalar_argument(name: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"the {name!r} quantity takes one finite scalar argument") from exc
    if not np.isfinite(number):
        raise ValueError(f"the {name!r} quantity takes one finite scalar argument")
    return number


def _validated_probability(name: str, value: float) -> float:
    if not 0.0 < value < 1.0:
        raise ValueError(f"the {name!r} quantity needs a probability strictly inside (0, 1)")
    return value


def _distribution_quantity(
    family: Any, name: str, value: float, weights: NDArray[np.float64] | None
) -> Callable[[NDArray], NDArray[np.float64]]:
    def _levels(theta: NDArray, level: float) -> NDArray[np.float64]:
        return np.full(len(theta), level, dtype=np.float64)

    if name == "expected_shortfall":
        _validated_probability(name, value)
        if weights is None:
            if not isinstance(family, ExpectedShortfallFamily):
                raise NotImplementedError(
                    f"{type(family).__name__} has no certified expected shortfall; "
                    "implement ExpectedShortfallFamily"
                )
            return lambda theta: np.asarray(
                family.expected_shortfall(_levels(theta, value), theta), dtype=np.float64
            )
        if not isinstance(family, PriorWeightedExpectedShortfallFamily):
            raise NotImplementedError(
                _PRIOR_WEIGHT_SHORTFALL_REFUSAL.format(family=type(family).__name__)
            )
        return lambda theta: np.asarray(
            family.expected_shortfall_prior_weighted(_levels(theta, value), theta, weights),
            dtype=np.float64,
        )

    if weights is None:
        if not isinstance(family, DistributionFunctionFamily):
            raise NotImplementedError(
                f"the {name!r} quantity needs a family with a cdf and a quantile; "
                "this one has neither"
            )
        cdf = family.cdf
        quantile = family.quantile
    else:
        # The row's own law, never the unit one: a family that cannot state it
        # refuses rather than silently reporting the unweighted quantity.
        if not isinstance(family, PriorWeightedDistributionFunctionFamily):
            raise NotImplementedError(_PRIOR_WEIGHT_REFUSAL.format(family=type(family).__name__))

        def cdf(y: NDArray, theta: NDArray) -> NDArray:
            return family.cdf_prior_weighted(y, theta, weights)

        def quantile(p: NDArray, theta: NDArray) -> NDArray:
            return family.quantile_prior_weighted(p, theta, weights)

    if name == "cdf":
        return lambda theta: np.asarray(cdf(_levels(theta, value), theta), dtype=np.float64)
    if name == "exceedance":
        return lambda theta: 1.0 - np.asarray(cdf(_levels(theta, value), theta), dtype=np.float64)

    _validated_probability(name, value)
    if name == "quantile":
        return lambda theta: np.asarray(quantile(_levels(theta, value), theta), dtype=np.float64)
    raise ValueError(f"unsupported distribution quantity {name!r}")


def resolve_quantity(
    family: Any, quantity: Quantity, *, weights: NDArray | None = None
) -> Callable[[NDArray], NDArray[np.float64]]:
    """Resolve a quantity name into a map from ``theta (m, k)`` to ``(m,)`` values.

    ``"mean"`` is the family's default prediction; ``("parameter", name)`` one
    natural parameter; ``("quantile", p)``, ``("cdf", y)``, ``("exceedance", t)``
    and ``("expected_shortfall", p)`` come from the family's distribution
    functions.  A callable passes through unchanged.  The threshold of ``cdf``
    and ``exceedance`` is a scalar: a row-varying threshold is not expressible
    through a map that sees only ``theta``, and belongs in a callable.

    ``weights`` are the rows' prior weights, which are part of each row's own
    law -- a Gaussian variance ``sigma^2 / w``, a gamma shape and scale in
    ``w``, a Tweedie rate and compound scale in ``w`` -- so the distribution
    functions above read the prior-weighted pair instead, and a family without
    one refuses.  They must give one weight per row of the ``theta`` the
    returned map is called on.  ``"mean"`` and ``("parameter", name)`` do not
    read them: the prior weight leaves the mean and the natural parameters of
    the row law alone.
    """
    if callable(quantity):
        return cast("Callable[[NDArray], NDArray[np.float64]]", quantity)
    if isinstance(quantity, str):
        name, arguments = quantity, ()
    elif isinstance(quantity, tuple) and quantity and isinstance(quantity[0], str):
        name, arguments = quantity[0], tuple(quantity[1:])
    else:
        raise ValueError(f"unsupported posterior quantity {quantity!r}")

    if name == "mean" and not arguments:
        if not isinstance(family, DefaultPredictionFamily):
            raise NotImplementedError(
                "this family has no default prediction; name a parameter or pass a callable"
            )
        return lambda theta: np.asarray(family.default_prediction(theta), dtype=np.float64)

    if name == "parameter" and len(arguments) == 1:
        names = tuple(parameter.name for parameter in family.parameters)
        if arguments[0] not in names:
            raise ValueError(f"unknown parameter {arguments[0]!r}; the family has {names}")
        index = names.index(arguments[0])
        return lambda theta: np.asarray(theta, dtype=np.float64)[:, index]

    if name in ("quantile", "cdf", "exceedance", "expected_shortfall") and len(arguments) == 1:
        return _distribution_quantity(
            family, name, _scalar_argument(name, arguments[0]), _prior_weight_law(weights)
        )

    raise ValueError(f"unsupported posterior quantity {quantity!r}")


def posterior_bounds(
    fitted: Any,
    X: FrameLike | EagerFrame,
    quantity: Quantity,
    *,
    level: float = 0.9,
    n_draws: int = 1000,
    covariance: CovarianceKind = "fixed",
    seed: int = 42,
    draws: PosteriorDraws | None = None,
    offsets: Mapping[str, NDArray] | None = None,
    return_draws: bool = False,
    chunk_rows: int | None = None,
    weights: NDArray | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, NDArray[np.float64]]:
    """Summarise a quantity of the posterior per row.

    The frame carries ``estimate`` (the plug-in value at ``beta_hat``), ``mean``,
    ``sd`` and the equal-tailed ``lower``/``upper`` at ``level``.  With
    ``return_draws=True`` the ``(draws, rows)`` matrix of the quantity is
    returned beside it.

    ``weights`` are one prior weight per row of ``X``; where they are not all
    one, the distribution-function quantities read each row's own
    prior-weighted law (:func:`resolve_quantity`) and are chunked in step with
    the rows they belong to.
    """
    coverage = float(level)
    if not 0.0 < coverage < 1.0:
        raise ValueError("level must lie strictly inside (0, 1)")
    if draws is None:
        draws = posterior_draws(fitted, n_draws, covariance=covariance, seed=seed)

    frame = as_eager_frame(X)
    n_observations = len(frame)
    row_weights = _prior_weight_law(weights, n_observations)
    evaluate = resolve_quantity(fitted.family, quantity, weights=row_weights)
    estimate = np.asarray(
        evaluate(fitted.predict_parameters(frame, offsets=offsets)), dtype=np.float64
    )
    if estimate.shape != (n_observations,):
        raise ValueError("a posterior quantity must return one value per row")
    if not callable(quantity) and not np.all(np.isfinite(estimate)):
        raise ValueError(
            f"non-finite plug-in values from named posterior quantity {quantity!r}; "
            "posterior bounds are undefined"
        )

    tail = 0.5 * (1.0 - coverage)
    summaries = {
        name: np.empty(n_observations, dtype=np.float64)
        for name in ("mean", "sd", "lower", "upper")
    }
    quantity_draws = (
        np.empty((draws.n_draws, n_observations), dtype=np.float64) if return_draws else None
    )
    for rows, block in posterior_parameters(
        fitted, frame, draws, offsets=offsets, chunk_rows=chunk_rows
    ):
        width = block.shape[1]
        # ``block`` flattens draw-major, so the chunk's own weights tile in the
        # same order and every simulated row keeps the weight it was given.
        chunk = (
            evaluate
            if row_weights is None
            else resolve_quantity(
                fitted.family, quantity, weights=np.tile(row_weights[rows], draws.n_draws)
            )
        )
        values = np.asarray(chunk(block.reshape(-1, block.shape[2])), dtype=np.float64)
        if values.shape != (draws.n_draws * width,):
            raise ValueError("a posterior quantity must return one value per row")
        values = values.reshape(draws.n_draws, width)
        if not callable(quantity) and not np.all(np.isfinite(values)):
            raise ValueError(
                f"non-finite draws from named posterior quantity {quantity!r}; "
                "posterior bounds are undefined"
            )
        summaries["mean"][rows] = values.mean(axis=0)
        summaries["sd"][rows] = values.std(axis=0, ddof=1)
        summaries["lower"][rows], summaries["upper"][rows] = np.quantile(
            values, [tail, 1.0 - tail], axis=0
        )
        if quantity_draws is not None:
            quantity_draws[:, rows] = values

    bounds = pd.DataFrame({"estimate": estimate, **summaries}, index=_row_index(X, n_observations))
    if return_draws:
        assert quantity_draws is not None
        return bounds, quantity_draws
    return bounds


def _resolved_reduce(reduce: Any) -> Callable[[NDArray], NDArray] | None:
    if reduce is None or callable(reduce):
        return reduce
    if reduce == "sum":
        return lambda block: block.sum(axis=1)
    raise ValueError("reduce must be None, 'sum', or a callable over a (draws, rows) block")


def _theta_blocks(
    fitted: Any,
    frame: EagerFrame,
    *,
    n_draws: int,
    parameter_uncertainty: bool,
    covariance: CovarianceKind,
    seed: int,
    offsets: Mapping[str, NDArray] | None,
    chunk_rows: int,
) -> Generator[tuple[slice, NDArray[np.float64]], None, None]:
    """Yield ``(row_slice, theta)`` with ``theta`` flattened to ``(draws * rows, k)``."""
    if parameter_uncertainty:
        draws = posterior_draws(fitted, n_draws, covariance=covariance, seed=seed)
        for rows, block in posterior_parameters(
            fitted, frame, draws, offsets=offsets, chunk_rows=chunk_rows
        ):
            yield rows, block.reshape(-1, block.shape[2])
        return
    theta = np.asarray(fitted.predict_parameters(frame, offsets=offsets), dtype=np.float64)
    for start in range(0, len(theta), chunk_rows):
        stop = min(start + chunk_rows, len(theta))
        yield slice(start, stop), np.tile(theta[start:stop], (n_draws, 1))


def posterior_predictive(
    fitted: Any,
    X: FrameLike | EagerFrame,
    n_draws: int = 200,
    *,
    parameter_uncertainty: bool = True,
    reduce: Any = None,
    offsets: Mapping[str, NDArray] | None = None,
    seed: int = 42,
    covariance: CovarianceKind = "fixed",
    chunk_rows: int | None = None,
    weights: NDArray | None = None,
) -> NDArray[np.float64]:
    """Simulate responses through the family's quantile function.

    ``parameter_uncertainty=True`` draws ``theta`` from the posterior per draw;
    ``False`` repeats the plug-in ``theta_hat``, which is the envelope the
    plug-in Q-Q construction uses.  ``reduce`` maps each ``(draws, rows)`` block
    so a portfolio or segment aggregate never materialises ``(draws, rows)``
    over all rows: it must be **additive across row chunks** -- sums and counts
    qualify, means do not -- and ``reduce="sum"`` is the named convenience.  A
    reduce returning ``(draws,)`` is summed across chunks; one returning
    ``(draws, columns)`` is concatenated along the column axis.

    ``weights`` are one prior weight per row of ``X``.  A prior weight is part
    of the row's own law, so where they are not all one the responses are drawn
    through the family's ``quantile_prior_weighted`` -- a policy at a fifth of
    a year's exposure is simulated on its own law, not on a full year's -- and
    a family without that law refuses rather than simulating the unit one.
    """
    frame = as_eager_frame(X)
    row_weights = _prior_weight_law(weights, len(frame))
    family = fitted.family
    if row_weights is None:
        if not isinstance(family, DistributionFunctionFamily):
            raise NotImplementedError(
                "predictive draws need a family with a quantile function; this one has none"
            )
    elif not isinstance(family, PriorWeightedDistributionFunctionFamily):
        raise NotImplementedError(_PRIOR_WEIGHT_REFUSAL.format(family=type(family).__name__))
    count = int(n_draws)
    if count < 1:
        raise ValueError("n_draws must be at least 1")
    combine = _resolved_reduce(reduce)

    rows_per_chunk = _resolved_chunk_rows(
        chunk_rows, n_draws=count, n_parameters=len(fitted.layout.predictors)
    )
    # The coefficient draws consume ``seed`` directly; the uniforms take an
    # independent child of the same seed, so both are reproducible and neither
    # reuses the other's stream.
    uniform_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[1])

    pieces: list[NDArray[np.float64]] = []
    for rows, theta in _theta_blocks(
        fitted,
        frame,
        n_draws=count,
        parameter_uncertainty=parameter_uncertainty,
        covariance=covariance,
        seed=seed,
        offsets=offsets,
        chunk_rows=rows_per_chunk,
    ):
        width = rows.stop - rows.start
        probabilities = uniform_rng.uniform(
            _PROBABILITY_MARGIN, 1.0 - _PROBABILITY_MARGIN, size=(count, width)
        )
        # ``theta`` arrives draw-major, so the chunk's own weights tile in the
        # same order and every simulated row is drawn on its own law.
        levels = probabilities.reshape(-1)
        drawn = (
            family.quantile(levels, theta)
            if row_weights is None
            else family.quantile_prior_weighted(levels, theta, np.tile(row_weights[rows], count))
        )
        simulated = np.asarray(drawn, dtype=np.float64).reshape(count, width)
        pieces.append(simulated if combine is None else np.asarray(combine(simulated)))

    if not pieces:
        if combine is not None:
            raise ValueError("reduce needs at least one row to reduce")
        return np.empty((count, 0), dtype=np.float64)
    if combine is not None and all(piece.shape == (count,) for piece in pieces):
        return np.sum(pieces, axis=0, dtype=np.float64)
    if any(piece.ndim < 2 for piece in pieces):
        raise ValueError(
            "reduce must return a (draws,) summary that is additive across row chunks "
            "or a (draws, columns) block; it returned neither on every chunk"
        )
    return np.concatenate(pieces, axis=1)


def simultaneous_critical_value(
    grid_design: NDArray,
    coefficient_slice: slice | NDArray[np.intp],
    draws: PosteriorDraws | NDArray,
    beta_hat: NDArray,
    se: NDArray,
    *,
    alpha: float = 0.05,
) -> float:
    """Return the max-deviation critical value over the grid.

    The ``(1 - alpha)`` quantile of ``max_j |G (beta_d - beta_hat)|_j / s_j``
    over the posterior draws (Ruppert, Wand and Carroll 2003, section 6.5).  It
    is always at least the pointwise normal value, and one routine serves term
    effects and risk curves alike.
    """
    coefficients = (
        draws.coefficients
        if isinstance(draws, PosteriorDraws)
        else np.asarray(draws, dtype=np.float64)
    )
    if coefficients.ndim != 2:
        raise ValueError("draws must be a (draws, coefficients) matrix")
    grid = np.asarray(grid_design, dtype=np.float64)
    if grid.ndim != 2:
        raise ValueError("grid_design must be a (grid points, coefficients) matrix")
    errors = np.asarray(se, dtype=np.float64)
    if errors.shape != (grid.shape[0],):
        raise ValueError("se must give one standard error per grid point")
    probability = float(alpha)
    if not 0.0 < probability < 1.0:
        raise ValueError("alpha must lie strictly inside (0, 1)")

    centred = (
        coefficients[:, coefficient_slice]
        - np.asarray(beta_hat, dtype=np.float64)[coefficient_slice]
    )
    if centred.shape[1] != grid.shape[1]:
        raise ValueError("grid_design columns must match the sliced coefficients")
    deviation = np.abs(centred @ grid.T) / np.maximum(errors, _MINIMUM_STANDARD_ERROR)
    simulated = float(np.quantile(deviation.max(axis=1), 1.0 - probability))
    pointwise = float(stats.norm.isf(0.5 * probability))
    return float(np.maximum(simulated, pointwise))


__all__ = [
    "PosteriorDraws",
    "posterior_bounds",
    "posterior_covariance",
    "posterior_draws",
    "posterior_parameters",
    "posterior_predictive",
    "resolve_quantity",
    "simultaneous_critical_value",
]
