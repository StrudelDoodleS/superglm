# Adding a distributional family

A distributional family is a small adapter between the generic `SuperLSS`
machinery and a family-owned likelihood. The family declares an ordered set of
natural parameters, binds the response and resolved weights into an immutable
plan, initializes those parameters, and evaluates row likelihood derivatives.
The fixed-fit, REML/EFS, inference, and public API layers stay family-neutral.

Do not infer a family's parameter names or arity from its class suffix. `LS`
and `LSS` follow established model names, while the ordered `family.parameters`
tuple is the executable contract. The built-ins currently expose:

- two parameters: `GaussianLS`, `GammaLS`, `LogNormalLS`,
  `NegativeBinomialLS`, and `GeneralizedParetoLSS`;
- three parameters: `TweedieLSS`, `GeneralizedGammaLSS`,
  `TwoPieceLogNormalLSS`, and `TwoPieceNormalLSS`.

## Responsibility map

| Location | Responsibility |
| --- | --- |
| `src/superglm/distributional/family.py` | Own the shared structural protocols, parameter and observation metadata, initialization/evaluation records, and family-neutral validation. It does not own a particular distribution's formulas. |
| `src/superglm/distributional/weights.py` | Resolve and certify prior or frequency weights, their provenance, retained rows, and slicing identity. A family consumes `ResolvedLikelihoodWeights`; it does not reinterpret raw weights. |
| `src/superglm/distributional/families/<name>.py` | Own the family adapter: parameter metadata, observation admission, the bound likelihood plan, initialization, `to_config()`, translation to and from a numerical kernel, and any optional structural protocols the family can support. |
| `src/superglm/distributional/kernels/<name>.py` (optional) | Own primitive row numerics, local numerical-domain checks, a local weight-semantics type, and an immutable kernel result. It imports no distributional contract or aggregate namespace; a dedicated sibling numerical backend is the only permitted package-internal dependency. |
| `src/superglm/distributional/serialization.py` | Own exact artifact codec registration and manifest validation only for families that explicitly promise persistence. An unregistered family is refused before pickle serialization. |
| `tests/` | Prove the family law against independent row and derivative oracles, weight identities, initialization and refusal boundaries, packed-channel order, public fitting, optional behavior, persistence ordering when promised, and import boundaries. |

## Solver-facing adapter contract

The four-method `DistributionalFamily` interface is the solver-facing adapter
contract. Its current public signature is:

```python
from typing import Protocol, runtime_checkable

from numpy.typing import NDArray

from superglm.distributional import (
    FamilyLikelihoodPlan,
    InitialParameterState,
    NaturalLikelihoodEvaluation,
    ObservationContract,
    ParameterSpec,
    ResolvedLikelihoodWeights,
)


@runtime_checkable
class DistributionalFamily(Protocol):
    @property
    def parameters(self) -> tuple[ParameterSpec, ...]: ...

    def bind_likelihood(
        self,
        y: NDArray,
        weights: ResolvedLikelihoodWeights,
        observation: ObservationContract,
    ) -> FamilyLikelihoodPlan: ...

    def initialize(
        self,
        y: NDArray,
        plan: FamilyLikelihoodPlan,
    ) -> InitialParameterState: ...

    def evaluate_natural(
        self,
        y: NDArray,
        theta: NDArray,
        plan: FamilyLikelihoodPlan,
        *,
        derivative_order: int = 2,
    ) -> NaturalLikelihoodEvaluation: ...
```

`parameters` fixes the natural-parameter order used by predictors, scores, and
curvature channels. `bind_likelihood()` owns the resolved weights in a plan;
`take()` on that plan must preserve their certified row identity.
`initialize()` returns a finite `n × k` natural-parameter matrix.
`evaluate_natural()` returns weighted row likelihood values and exactly the
requested derivatives. A derivative order of zero has neither score nor
Hessian, order one has a score but no Hessian, and order two has both. Values
other than an integer from zero through two, including booleans, are refused.

For `k` parameters, `hessian_packed` has width `k * (k + 1) // 2`. It is the
raw signed Hessian of the weighted per-row optimizing log likelihood, in
canonical upper-triangular order:

```text
(0, 0), (0, 1), ..., (0, k - 1), (1, 1), ..., (k - 1, k - 1)
```

It is not negated curvature or Fisher information. For four parameters the ten
channels are `(a,a), (a,b), (a,c), (a,d), (b,b), (b,c), (b,d), (c,c),
(c,d), (d,d)`, so the diagonal indices are `(0, 4, 7, 9)`.

## Complete-fit configuration

Complete fitting has a separate structural metadata requirement:

```python
from collections.abc import Mapping
from typing import Protocol, runtime_checkable


@runtime_checkable
class ConfigurableDistributionalFamily(Protocol):
    def to_config(self) -> Mapping[str, object]: ...
```

`to_config()` must return a nonempty mapping whose keys are nonempty strings.
The mapping should completely identify configuration that can change the
family's fitted law. Current null fitting deep-copies and snapshots this
mapping as read-only fit metadata, so later caller mutation cannot rewrite the
accepted state. Consequently, `ConfigurableDistributionalFamily` metadata is
required for a complete fit even when artifact persistence is neither needed
nor supported.

This configuration is not a serialization codec. In particular, a usable
custom family does not become a registered artifact type merely by returning
`{"type": "MyFamily"}`.

## Optional structural protocols

Optional behavior is detected structurally with the public runtime-checkable
protocols below. Do not add family-name branches to a solver or facade.

| Protocol | Member and responsibility | Behavior when absent |
| --- | --- | --- |
| `LikelihoodPlanValidatingFamily` | `validate_likelihood_plan(y, plan)` performs one-shot family-owned plan validation and returns the canonical, finite, exact-shape, read-only `float64` response. | Fixed fitting validates the plan's structural invariants and freezes the supplied finite response itself. |
| `ExpectedInformationFamily` | `expected_information_natural(theta, plan)` returns natural-scale Fisher information. It is the fallback when the terminal observed curvature is materially indefinite, and `SuperLSS(coefficient_curvature="fisher")` requests it for the whole solve. | The default coefficient solve is the same: Newton on the observed Hessian. `SuperLSS(coefficient_curvature="fisher")` refuses at construction, a route explicitly requesting Fisher chunking refuses, and repeated material indefiniteness of the terminal observed curvature refuses the fit instead of falling back. |
| `PredictorCurvatureDirectionalFamily` | `predictor_curvature_directional_derivative(y, eta, eta_direction, links, plan)` supplies the exact directional derivative of `curvature_packed` (the negated predictor-scale Hessian, packed upper-triangular) along `eta_direction`. | The engine differences the family's own order-two evaluation along the unit direction (Richardson order four, link-scale step 1e-3) and carries an error certificate into the endpoint decision band; the evidence is labelled `finite-difference-curvature-direction/v1`. Implement the analytic method only when you need `matched_certified`. |
| `DefaultPredictionFamily` | `default_prediction_name` names the response quantity and `default_prediction(theta)` computes it. | `predict_parameters()` remains available; `predict()` raises `NotImplementedError` and tells the caller to use `predict_parameters()`. |
| `DistributionFunctionFamily` | `cdf(y, theta)` and `quantile(p, theta)` row-wise from natural parameters; backs `predict_cdf()` / `predict_quantile()`. | Both facade methods raise `NotImplementedError` naming `predict_parameters()`. |
| `FitFailureDiagnosingFamily` | `diagnose_repeated_curvature_failure(y, weights)` may translate a repeated generic curvature failure into a family-specific exception. | The original repeated-curvature exception is preserved. A missing, failing, or non-exception diagnosis never hides it. |
| `PriorWeightedDistributionFunctionFamily` | `cdf_prior_weighted(y, theta, weights)` and `quantile_prior_weighted(p, theta, weights)` give the row law when a prior weight is part of it — the Gaussian variance `sigma^2 / w`, the gamma shape and scale, the Tweedie dispersion `phi / w`. | Every caller that needs a row law under non-unit prior weights refuses and names the family, rather than quietly reading the unit-weight distribution function. Residuals, calibration, predictive draws and posterior bounds are all affected; unit prior weights and frequency semantics are unaffected. |
| `ExpectedShortfallFamily` | `expected_shortfall(p, theta)` gives the certified row-wise upper conditional tail mean for an interior probability. This is independent of distribution-function support: generic quantile quadrature is not a certified tail mean. | The named `("expected_shortfall", p)` posterior quantity refuses and asks for this protocol. |
| `PriorWeightedExpectedShortfallFamily` | `expected_shortfall_prior_weighted(p, theta, weights)` gives the same tail mean when a non-unit prior weight changes the row law. | Non-unit prior-weighted expected shortfall refuses rather than substituting the unit law. Unit weights take the unweighted path exactly. |
| `AtomFamily` | `cdf_left_limit(y, theta, weights=None)` gives `P(Y < y)`, which is the bottom of the jump at a point mass. | The probability-integral transform is read as continuous. For a family with an atom that silently makes the transform non-uniform on the atom's rows, so a family with a point mass must implement it. |
| `VarianceFamily` | `variance(theta)` gives `Var(Y \| theta)` in closed form at unit prior weight. | The actual-versus-expected table simulates the variance from plug-in predictive draws instead and records `variance_law` as `"draws"`. |
| `PriorWeightedVarianceFamily` | `variance_prior_weighted(theta, weights)` gives the same second moment when a prior weight is inside the row's law. | The table falls back to simulation on the weighted law where the family has one (`"prior_weighted_draws"`), and only then to the unit-weight variance (`"family_unit_law"`), which ignores the weight and says so. |

Implementing `ExpectedInformationFamily` does not change how a family is
fitted by default; it adds the indefiniteness fallback and the opt-in Fisher
scoring request.

Unit-law and prior-weighted capabilities are separate on purpose.
`LogNormalLS`, for example, is not reproductive and refuses non-unit prior
weights, but still implements its unit-law distribution function, variance and
expected shortfall. `GaussianLS`, `GammaLS` and `TweedieLSS` implement both the
unit and prior-weighted distribution-function and variance protocols.

Expected-shortfall support is narrower and always family-owned through
`ExpectedShortfallFamily`. Its certified unit-law implementations are
`GaussianLS`, `GammaLS`, `LogNormalLS`, `GeneralizedGammaLSS` and
`GeneralizedParetoLSS`; only Gaussian and gamma also implement
`PriorWeightedExpectedShortfallFamily`. `TweedieLSS`, both two-piece families
and `NegativeBinomialLS` refuse expected shortfall. Protocol membership states
structural availability, not that every representable tail can be resolved:
Gamma, log-normal and generalized-gamma rows refuse when their float64 tail
certificate fails. Generalized-gamma location form returns `+∞` when the upper
first moment does not exist; mean form rejects that cell because a finite mean
parameter would be outside its model. Named posterior intervals refuse a
non-finite plug-in value or any non-finite posterior draw instead of reducing
such values into an interval.

CDF query support is distinct from response admission. The positive continuous
families return zero for every finite threshold `y ≤ 0`, including mixed
inside/outside vector calls, while `bind_likelihood()` keeps enforcing the
family's declared response support. Tweedie's value at zero is instead its
atom. A runtime-checkable protocol checks every member it declares, so folding
a weighted method into its unit-law protocol would silently remove the unit
capability from a non-reproductive family.

The generic score layer follows the resolved weight contract. Prior semantics
changes the row law and aggregates over retained physical rows. Frequency
semantics uses the unit law and literal replication mass for CRPS,
threshold-weighted CRPS, Murphy curves, paired standard errors and default
thresholds as well as log score. Zero-weight rows are omitted, and a comparison
with mixed semantics refuses whenever any retained weight is not exactly one.

`GammaLS` and `NegativeBinomialLS` own an exact response and a
response-dependent carrier in their plans. They therefore implement
`LikelihoodPlanValidatingFamily` and return the plan's canonical read-only
response after checking it against the supplied response. Gaussian and Tweedie
plans own no exact response requiring such a hook, so `GaussianLS` and
`TweedieLSS` rely on the generic response snapshot. Plan admission is
structural rather than exact-lineage based: subclasses of every plan are
admitted when their field invariants hold.

`PredictorCurvatureDirectionalFamily` has a smoothness precondition that no
amount of algebra removes: the row log-likelihood must be `C³` in every
natural parameter over the whole support the links can reach, because the
hook is the third derivative contracted with a predictor direction. The
Gaussian, gamma, negative-binomial, Tweedie, log-normal, generalized gamma and
generalized Pareto laws qualify. The two-piece laws do not: their curvature
jumps at the mode, so the true LAML gradient carries the density at the kink
times that jump, which a pointwise third derivative drops. For such a family
the differenced route with its certificate is the correct method and the hook
must not be implemented. The derivative orders each stage of the fit needs,
and which family supplies which, are tabulated in
[Derivative orders, and which families supply them](models/distributional.md#derivative-orders-and-which-families-supply-them).

## Primitive kernel rule

A family whose formulas merit a separate numerical module uses a primitive
kernel shaped like this:

```python
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

WeightSemantics = Literal["prior", "frequency"]


@dataclass(frozen=True)
class ExampleKernelEvaluation:
    optimizing_log_likelihood: NDArray[np.float64]
    score: NDArray[np.float64] | None
    hessian_packed: NDArray[np.float64] | None
    valid: NDArray[np.bool_]


def evaluate_example_rows(
    response: NDArray,
    first_parameter: NDArray,
    second_parameter: NDArray,
    weights: NDArray,
    semantics: WeightSemantics,
    *,
    derivative_order: int,
) -> ExampleKernelEvaluation:
    ...
```

`WeightSemantics` is local to the kernel. The result type is also kernel-owned
and must own read-only output arrays. The primitive module imports numerical
libraries and standard-library types, but no distributional contract, family
adapter, or aggregate namespace. A dedicated sibling numerical backend is the
only exception. Its family adapter passes primitive arrays and the resolved
semantics directly to this numerical owner, then constructs the public
`NaturalLikelihoodEvaluation`. Application and extension code should use the
public family contracts, not import private kernel modules.

## Worked four-parameter family

This complete public-only example is the extensibility regression. It supplies
four natural parameters and therefore ten packed Hessian channels, explicitly
admits only the complete observation contract, and provides nonempty
complete-fit configuration:

```python
from dataclasses import dataclass

import numpy as np
import pandas as pd

from superglm import SuperLSS
from superglm.distributional import (
    COMPLETE_OBSERVATION,
    FamilyLikelihoodPlan,
    InitialParameterState,
    NaturalLikelihoodEvaluation,
    ObservationContract,
    ParameterSpec,
    ParameterSupport,
    Predictor,
    ResolvedLikelihoodWeights,
    UnsupportedLikelihoodContractError,
)
from superglm.links import IdentityLink


def parameter(name: str) -> ParameterSpec:
    return ParameterSpec(
        name=name,
        default_link=IdentityLink(),
        role=name,
        support=ParameterSupport(),
        curvature="observed",
    )


@dataclass(frozen=True)
class FourParameterPlan:
    weights: ResolvedLikelihoodWeights

    @property
    def plan_identifier(self) -> str:
        return f"four-parameter/v1:{self.weights.digest}"

    def take(self, indices: np.ndarray) -> "FourParameterPlan":
        return FourParameterPlan(self.weights.take(indices))


@dataclass(frozen=True)
class FourParameterFamily:
    parameters = tuple(parameter(name) for name in ("a", "b", "c", "d"))

    def to_config(self) -> dict[str, object]:
        return {"type": "FourParameterFamily", "parameters": 4}

    @staticmethod
    def targets(y: np.ndarray) -> np.ndarray:
        return np.column_stack(
            (
                y,
                np.full(len(y), 0.25),
                np.full(len(y), -0.5),
                np.full(len(y), 1.0),
            )
        )

    def bind_likelihood(
        self,
        y: np.ndarray,
        weights: ResolvedLikelihoodWeights,
        observation: ObservationContract,
    ) -> FourParameterPlan:
        response = np.asarray(y, dtype=np.float64)
        if observation != COMPLETE_OBSERVATION:
            raise UnsupportedLikelihoodContractError("complete observations are required")
        if response.shape != weights.values.shape or not np.all(np.isfinite(response)):
            raise ValueError("response and resolved weights must contain the same finite rows")
        return FourParameterPlan(weights)

    def initialize(
        self,
        y: np.ndarray,
        plan: FamilyLikelihoodPlan,
    ) -> InitialParameterState:
        if not isinstance(plan, FourParameterPlan):
            raise TypeError("FourParameterFamily requires FourParameterPlan")
        return InitialParameterState(self.targets(np.asarray(y, dtype=np.float64)))

    def evaluate_natural(
        self,
        y: np.ndarray,
        theta: np.ndarray,
        plan: FamilyLikelihoodPlan,
        *,
        derivative_order: int = 2,
    ) -> NaturalLikelihoodEvaluation:
        if not isinstance(plan, FourParameterPlan):
            raise TypeError("FourParameterFamily requires FourParameterPlan")
        if (
            isinstance(derivative_order, bool | np.bool_)
            or not isinstance(derivative_order, int | np.integer)
            or int(derivative_order) not in (0, 1, 2)
        ):
            raise ValueError("derivative_order must be an integer from zero through two")
        order = int(derivative_order)
        targets = self.targets(np.asarray(y, dtype=np.float64))
        residual = targets - np.asarray(theta, dtype=np.float64)
        weights = plan.weights.values
        score = None if order == 0 else weights[:, None] * residual
        hessian = None
        if order == 2:
            hessian = np.zeros((len(targets), 10), dtype=np.float64)
            hessian[:, (0, 4, 7, 9)] = -weights[:, None]
        return NaturalLikelihoodEvaluation(
            optimizing_log_likelihood=-0.5 * weights * np.sum(residual * residual, axis=1),
            parameter_independent_carrier=np.zeros(len(targets)),
            score=score,
            hessian_packed=hessian,
            valid=np.ones(len(targets), dtype=bool),
        )


frame = pd.DataFrame({"row": np.linspace(-1.0, 1.0, 12)})
response = np.linspace(-0.8, 1.2, len(frame))
model = SuperLSS(
    family=FourParameterFamily(),
    predictors=tuple(Predictor(name, {}) for name in ("a", "b", "c", "d")),
).fit(frame, response, lambdas={})
parameters = model.predict_parameters(frame)
```

The nonempty `to_config()` metadata is load-bearing: this family needs it to
complete null fitting. It can nevertheless complete the public fixed fit and
call `predict_parameters()` without implementing `DefaultPredictionFamily` or
registering a persistence codec. `model.predict(frame)` produces the
actionable default-prediction refusal and directs callers to
`predict_parameters()`. `model.to_bytes()` produces the actionable
unregistered-family persistence refusal, `only built-in families can be
serialized`. No artifact round trip belongs in this example because the
family does not promise persistence.

## Persistence is explicit

Family configuration and artifact registration solve different problems.
`to_config()` records the configuration of an accepted complete fit. An exact
codec registration additionally pins the concrete Python class, qualified
Python type, configuration type and schema, validation, and reconstruction
rules needed to trust a stored model.

Serialization accepts only exact registered family classes. An unregistered
family, including a subclass of a registered built-in, is refused while the
manifest is being selected, before family metadata is encoded and before the
model reaches pickle serialization. Deserialization admits only an exact
registered `(python_type, config.type)` pair and validates it before base64
decoding or unpickling. These ordering guarantees keep corrupted or unknown
artifacts away from executable payload handling.

Complete fitting therefore does not imply a storage codec. Add an exact codec
entry only when the family explicitly promises a compatible, validated
artifact lifecycle; otherwise fitting and `predict_parameters()` are the
supported extension surface and `to_bytes()` refuses safely.

## Evidence checklist

Before proposing a package-provided family, record all applicable evidence:

- [ ] Configuration admission rejects a missing or non-callable `to_config()`,
  an empty/non-mapping result, and non-string or empty keys; snapshot tests
  prove caller mutation cannot change accepted metadata.
- [ ] An independent row likelihood and derivative oracle covers ordinary and
  adversarial natural-parameter points without reusing production formulas.
- [ ] Derivative-order laziness proves order zero omits score and Hessian,
  order one omits Hessian, order two supplies both, and invalid orders refuse.
- [ ] Prior/frequency identities prove the documented prior-weight law and
  literal frequency-row replication, including every response carrier and the
  score, comparison, standard-error and default-threshold reductions that read
  those semantics; zero rows and mixed-semantics refusals are covered.
- [ ] Initialization/refusal tests prove finite supported starts and named,
  deterministic domain refusals rather than silent clipping or fallback.
- [ ] Packed width is exactly `k * (k + 1) // 2`, with every score and Hessian
  channel checked in canonical upper-triangular order.
- [ ] A public fixed fit through `SuperLSS` proves predictor ordering,
  parameter prediction, and generic curvature-channel telemetry without an
  engine family branch.
- [ ] Optional protocols are tested both when present and absent, including
  default prediction, distribution functions, expected shortfall and its
  prior-weighted variant, plan validation, Fisher information, directional
  curvature, and failure diagnosis.
- [ ] Positive continuous CDF tests distinguish a finite outside-support query
  (`y ≤ 0` gives zero) from likelihood response admission, and atom families
  retain their jump at zero.
- [ ] Serialization tamper ordering proves exact class/config-pair admission
  and refusal before payload decoding, unpickling, or pickling as appropriate.
- [ ] AST import checks keep public integration tests on documented namespaces
  and keep primitive kernels free of reverse distributional imports.
- [ ] Ruff check and format checks pass for production and test code.
- [ ] Focused family, kernel, public API, and persistence tests pass before any
  broader suite is used as integration evidence.

## Change checklist

For a normal package-provided family, production changes are limited to the
family adapter with `to_config()`, an optional primitive kernel, its public
export, and an exact serialization codec entry only if persistence is
promised. External structural families need no production registration merely
to fit.

Keep each added or materially changed function within the ordinary ≤60-line
target. Give one lifecycle per class. Do not introduce nested callback
choreography or a one-line forwarding ladder. Permit at most two pass-through
hops carrying substantially the same arguments, and have the family adapter
call its numerical owner directly.

Edits to fixed fitting, REML/EFS, inference, diagnostics, or API admission in
order to add a family indicate a boundary failure. Stop and request design
review instead of teaching those generic layers the new family's name.
