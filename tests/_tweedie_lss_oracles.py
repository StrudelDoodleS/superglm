"""Literal evidence for the internal Tweedie LSS kernel.

Independent mgcv/CRAN, analytic, and 100-decimal fixtures remain distinct from
the explicitly labelled frozen characterization below. This module deliberately
contains no imports from production Tweedie code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

WeightSemantics = Literal["prior", "frequency"]
OracleSource = Literal[
    "direct-mixture-100-digit",
    "frozen-mgcv-cran-binary64",
    "mgcv-1.9-3-ldTweedie-binary64",
    "analytic-zero-100-digit",
    "direct-recurrence-100-digit",
]

CASE_IDS = (
    "lower-positive-light-prior",
    "quarter-power-zero-frequency",
    "quarter-power-tiny-positive-prior",
    "mid-left-unit-prior",
    "mid-left-unit-frequency",
    "mid-right-large-positive-prior",
    "three-quarter-positive-frequency",
)


@dataclass(frozen=True, slots=True)
class TweedieLSSOracleCase:
    """One immutable complete normalized row-likelihood oracle."""

    id: str
    y: float
    mean: float
    dispersion: float
    power: float
    semantics: WeightSemantics
    weight: float
    value: float
    score: tuple[float, float, float]
    hessian: tuple[float, float, float, float, float, float]
    source: OracleSource
    value_atol: float = 2.0e-12
    score_rtol: float = 2.0e-7
    score_atol: float = 2.0e-8
    hessian_rtol: float = 2.0e-5
    hessian_atol: float = 2.0e-6
    envelope_note: str = (
        "Initial reviewed binary64 envelope; the 37-log-unit evaluator is a "
        "deterministic point truncation, not a certified tail enclosure."
    )


TWEEDIE_LSS_CASES = (
    TweedieLSSOracleCase(
        id="lower-positive-light-prior",
        y=1.8,
        mean=1.3,
        dispersion=0.6,
        power=1.05,
        semantics="prior",
        weight=0.4,
        value=-1.5150198182932293,
        score=(0.253068578338362, 6.758986238192257, -1.5777870702178323),
        hessian=(
            -0.7105387007192465,
            -0.42178096389727,
            -0.06639615141557609,
            -50.78297619194669,
            -185.28718407733157,
            3.666978829806396,
        ),
        source="frozen-mgcv-cran-binary64",
    ),
    TweedieLSSOracleCase(
        id="quarter-power-zero-frequency",
        y=0.0,
        mean=2.5,
        dispersion=1.4,
        power=1.25,
        semantics="frequency",
        weight=4.0,
        value=-7.574006940638578,
        score=(-2.2722020821915736, 5.410004957598985, -3.1586835579937844),
        hessian=(
            0.22722020821915737,
            1.6230014872796956,
            2.0819977088572967,
            -7.7285785108556935,
            2.256202541424132,
            -14.782206836013982,
        ),
        source="frozen-mgcv-cran-binary64",
    ),
    TweedieLSSOracleCase(
        id="quarter-power-tiny-positive-prior",
        y=0.015,
        mean=2.5,
        dispersion=1.4,
        power=1.25,
        semantics="prior",
        weight=0.4,
        value=-10.428077945053488,
        score=(-0.2258568869698424, -2.306404398154526, 68.90781964636697),
        hessian=(
            0.022040360197258266,
            0.16132634783560173,
            0.20695057226041522,
            1.2540471973228064,
            11.60631267691837,
            -563.9377055455428,
        ),
        source="direct-mixture-100-digit",
    ),
    TweedieLSSOracleCase(
        id="mid-left-unit-prior",
        y=1.1,
        mean=0.9,
        dispersion=0.3,
        power=1.49,
        semantics="prior",
        weight=1.0,
        value=-0.48619249762821295,
        score=(0.7799870673332027, -1.5282853694625564, -0.053466462266083665),
        hessian=(
            -5.191247259250981,
            -2.599956891110676,
            0.0821798396206619,
            3.964441942846567,
            -0.0025244737825009505,
            0.03405545329850115,
        ),
        source="direct-mixture-100-digit",
    ),
    TweedieLSSOracleCase(
        id="mid-left-unit-frequency",
        y=1.1,
        mean=0.9,
        dispersion=0.3,
        power=1.49,
        semantics="frequency",
        weight=1.0,
        value=-0.4861924976282132,
        score=(0.779987067333203, -1.528285369462547, -0.053466462266084584),
        hessian=(
            -5.191247259250981,
            -2.5999568911106756,
            0.0821798396206602,
            3.964441942847764,
            -0.002524473782416184,
            0.034055453298432216,
        ),
        source="frozen-mgcv-cran-binary64",
    ),
    TweedieLSSOracleCase(
        id="mid-right-large-positive-prior",
        y=8.0,
        mean=2.0,
        dispersion=2.2,
        power=1.51,
        semantics="prior",
        weight=3.25,
        value=-6.448153700850504,
        score=(3.112122167828517, 1.6392073333724384, 3.6472897053085442),
        hessian=(
            -2.8683392646819494,
            -1.4146009853765984,
            -2.1571587061884405,
            -1.6039343423245545,
            -2.1779228333857934,
            -5.953371116653521,
        ),
        source="frozen-mgcv-cran-binary64",
    ),
    TweedieLSSOracleCase(
        id="three-quarter-positive-frequency",
        y=3.4,
        mean=2.2,
        dispersion=1.1,
        power=1.75,
        semantics="frequency",
        weight=3.0,
        value=-6.728986295387481,
        score=(0.8235123644386292, -1.2485287236195795, -1.6633379608797408),
        hessian=(
            -1.3413269572295855,
            -0.7486476040351174,
            -0.6493043850926205,
            0.5948518483798107,
            -0.4140961971623639,
            0.12789091652431644,
        ),
        source="direct-mixture-100-digit",
    ),
)

assert tuple(case.id for case in TWEEDIE_LSS_CASES) == CASE_IDS


TWEEDIE_POWER_RANGE_ZERO_CASES = (
    TweedieLSSOracleCase(
        id="lower-power-boundary-zero-prior",
        y=0.0,
        mean=1.3,
        dispersion=0.6,
        power=1.05,
        semantics="prior",
        weight=0.4,
        value=-0.9003913629301712,
        score=(-0.6579783036797405, 1.5006522715502854, -0.711549864363703),
        hessian=(
            0.025306857833836174,
            1.0966305061329008,
            0.17262999368050258,
            -5.002174238500951,
            1.1859164406061717,
            -1.5599781604626553,
        ),
        source="analytic-zero-100-digit",
    ),
    TweedieLSSOracleCase(
        id="upper-power-boundary-zero-prior",
        y=0.0,
        mean=4.0,
        dispersion=0.8,
        power=1.95,
        semantics="prior",
        weight=0.25,
        value=-6.698584140851832,
        score=(-0.0837323017606479, 8.37323017606479, -124.68547339508662),
        hessian=(
            0.019886421668153878,
            0.10466537720080988,
            0.11607761777437528,
            -20.933075440161976,
            155.85684174385828,
            -5000.292355561293,
        ),
        source="analytic-zero-100-digit",
    ),
)


@dataclass(frozen=True, slots=True)
class TweedieRefusalCase:
    """A supported mixed-law row outside the certified executable power range."""

    id: str
    y: float
    mean: float
    dispersion: float
    power: float
    semantics: WeightSemantics
    weight: float
    source: Literal["frozen-mgcv-cran-binary64"] = "frozen-mgcv-cran-binary64"


TWEEDIE_OUTSIDE_POWER_RANGE_REFUSAL_CASES = (
    TweedieRefusalCase("lower-wall-zero-unit-prior", 0.0, 0.4, 0.7, 1.0101, "prior", 1.0),
    TweedieRefusalCase("lower-wall-positive-prior", 0.35, 0.8, 0.9, 1.0101, "prior", 3.0),
    TweedieRefusalCase("upper-wall-positive-unit-prior", 1.7, 1.4, 3.0, 1.9899, "prior", 1.0),
    TweedieRefusalCase("upper-wall-zero-prior", 0.0, 4.0, 0.8, 1.9899, "prior", 0.25),
)


@dataclass(frozen=True, slots=True)
class CenteredRhoMomentOracle:
    """High-mode p=1.5 recurrence oracle for centered nuisance moments.

    The expected values are infinite-series quantities.  A separate
    100-decimal sum stopped at the production 37-log-unit endpoints changes
    score-rho, rho variance, rho/p covariance, and p variance by at most
    ``5e-16``, ``1.17e-9``, ``2.33e-9``, and ``4.65e-9`` respectively.
    The float64 envelopes add explicit operation/conditioning allowances to
    those truncation effects; the rho-Hessian allowance remains below the
    retained ``1.875e-7`` signal.
    """

    y: float = 1.0
    mean: float = 1.0
    dispersion: float = 2.0e-6
    power: float = 1.5
    weight: float = 1.0
    expected_score_rho: float = -0.5000001875000938
    expected_hessian_rho_rho: float = -1.8750018750018457e-7
    expected_covariance_q_rho_p: float = -4000001.000000375
    expected_variance_q_p: float = 8000006.000002083
    score_rho_atol: float = 2.0e-9
    hessian_rho_rho_atol: float = 6.0e-8
    covariance_q_rho_p_atol: float = 1.0e-6
    variance_q_p_atol: float = 4.0e-6
    source: OracleSource = "direct-recurrence-100-digit"
    envelope_note: str = (
        "The p=1.5 series was summed by 100-digit Decimal adjacent-weight "
        "recurrence through relative mass 1e-90.  The envelopes add the "
        "measured 37-log endpoint effect to 4, 128, 1024, and 2048 binary64 "
        "epsilon-scaled cancellation factors for score-rho, rho variance, "
        "rho/p covariance, and p variance."
    )


CENTERED_RHO_MOMENT_ORACLE = CenteredRhoMomentOracle()


@dataclass(frozen=True, slots=True)
class TweedieScalarDensityOracle:
    """One independently evaluated normalized positive-row density."""

    id: str
    y: float
    mean: float
    dispersion: float
    power: float
    semantics: WeightSemantics
    weight: float
    value: float
    source: OracleSource
    value_atol: float


TWEEDIE_UPPER_POWER_RANGE_SCALAR_DENSITY_ORACLE = TweedieScalarDensityOracle(
    id="upper-power-boundary-positive-unit-prior",
    y=1.7,
    mean=1.4,
    dispersion=3.0,
    power=1.95,
    semantics="prior",
    weight=1.0,
    value=-2.220242196662273,
    source="mgcv-1.9-3-ldTweedie-binary64",
    value_atol=2.0e-12,
)


FrozenCharacterizationSource = Literal["frozen-python-characterization/v1"]
FrozenMirrorChannelSource = Literal["frozen-python-mirror/v1"]
FrozenCompiledTermsSource = Literal["frozen-numba-evaluation/v1"]
FROZEN_CHARACTERIZATION_PROVENANCE = (
    "Numerical channels were emitted by an independent Python mathematics mirror "
    "before its removal. Exact window counts, statuses, and failing rows record the "
    "paired compiled execution. These literals are frozen characterization, not "
    "external oracles."
)


@dataclass(frozen=True, slots=True)
class FrozenTweedieEvaluation:
    """A mirror-derived operational characterization, never an external oracle."""

    id: str
    rows: tuple[tuple[float, float, float, float, float], ...]
    semantics: WeightSemantics
    derivative_order: int
    max_terms: int
    log_cutoff: float
    log_likelihood: tuple[float, ...]
    score: tuple[tuple[float, float, float], ...] | None
    hessian: tuple[tuple[float, float, float, float, float, float], ...] | None
    terms: tuple[int, ...]
    source: FrozenCharacterizationSource = "frozen-python-characterization/v1"


FROZEN_ADVERSARIAL_EVALUATION = FrozenTweedieEvaluation(
    id="adversarial-grid-order-two",
    rows=(
        (1.0, 1.0, 2.0e-6, 1.5, 1.0),
        (0.5, 1.0, 1.0, 1.5, 1.0),
        (
            1.0000000000000002,
            1.0,
            1.0000000000000002,
            1.5000000000000002,
            1.0000000000000002,
        ),
        (1.8, 1.3, 0.6, 1.05, 0.4),
        (1.7, 1.4, 3.0, 1.95, 1.0),
    ),
    semantics="prior",
    derivative_order=2,
    max_terms=100_000,
    log_cutoff=37.0,
    log_likelihood=(
        5.642242967849597,
        -0.7403920975964723,
        -1.0286152203419816,
        -1.5150198182932155,
        -2.2202421966622126,
    ),
    score=(
        (0.0, -250000.0938307494, -5.798144186996979e-10),
        (-0.5, -0.5512711769290246, 0.5323589955355579),
        (2.220446049250313e-16, -0.6321890694374391, 0.00914777677677403),
        (0.25306857833836177, 6.758986238192234, -1.5777870702175392),
        (0.051886016619718006, -0.22878909000786388, -0.04085434975834801),
    ),
    hessian=(
        (
            -499999.9999999997,
            -0.0,
            -0.0,
            124999988242.0525,
            0.3413297235965729,
            -2.4437904357910156e-6,
        ),
        (-0.25, 0.5, 0.0, 0.042286558700213916, 0.6164794239453881, -0.14990168572584395),
        (
            -1.0000000000000004,
            -2.2204460492503126e-16,
            -0.0,
            0.4393916332966328,
            0.06501042564253633,
            0.3947029745320645,
        ),
        (
            -0.7105387007192466,
            -0.42178096389726966,
            -0.06639615141557793,
            -50.78297619194616,
            -185.28718407733598,
            3.666978829782977,
        ),
        (
            -0.24522319759557196,
            -0.017295338873239335,
            -0.01745820406140194,
            0.061503358433340476,
            0.10586889026782782,
            -0.008451822967344924,
        ),
    ),
    terms=(12_167, 14, 16, 4, 39),
)


FROZEN_HUGE_CAP_EVALUATION = FrozenTweedieEvaluation(
    id="huge-python-cap-success",
    rows=((1.1, 0.9, 0.3, 1.49, 1.0),),
    semantics="prior",
    derivative_order=2,
    max_terms=2**100,
    log_cutoff=37.0,
    log_likelihood=(-0.48619249762821254,),
    score=((0.7799870673332028, -1.5282853694625647, -0.05346646226606078),),
    hessian=(
        (
            -5.19124725925098,
            -2.599956891110676,
            0.08217983962066192,
            3.964441942846876,
            -0.0025244737827308703,
            0.034055453298741156,
        ),
    ),
    terms=(28,),
)


@dataclass(frozen=True, slots=True)
class FrozenTweedieCutoffCase:
    """Mirror channels and compiled terms at selected adjacent-float cutoffs."""

    id: str
    row: tuple[float, float, float, float, float]
    semantics: WeightSemantics
    cutoffs: tuple[float, float, float]
    terms: tuple[int, int, int]
    log_likelihood: float
    score: tuple[float, float, float]
    hessian: tuple[float, float, float, float, float, float]
    # Ordered as canonical b_pp, series mean_q_pp, series variance_q_p after
    # applying any frequency multiplier. Their exact sum is hessian[5].
    p_p_constituents: tuple[float, float, float]
    source: FrozenCharacterizationSource = "frozen-python-characterization/v1"
    channels_source: FrozenMirrorChannelSource = "frozen-python-mirror/v1"
    terms_source: FrozenCompiledTermsSource = "frozen-numba-evaluation/v1"


FROZEN_CUSTOM_CUTOFF_CASES = (
    FrozenTweedieCutoffCase(
        id="lower-positive-adjacent-cutoff",
        row=(1.8, 1.3, 0.6, 1.05, 0.4),
        semantics="prior",
        cutoffs=(38.586457943366945, 38.58645794336695, 38.58645794336696),
        terms=(4, 4, 4),
        log_likelihood=-1.5150198182932155,
        score=(0.25306857833836177, 6.758986238192234, -1.5777870702175392),
        hessian=(
            -0.7105387007192466,
            -0.42178096389726966,
            -0.06639615141557793,
            -50.78297619194616,
            -185.28718407733598,
            3.666978829782977,
        ),
        p_p_constituents=(-19201.552824924278, 19030.321550716082, 174.8982530379788),
    ),
    FrozenTweedieCutoffCase(
        id="tiny-positive-adjacent-cutoff",
        row=(0.015, 2.5, 1.4, 1.25, 0.4),
        semantics="prior",
        cutoffs=(38.03041909503295, 38.030419095032954, 38.03041909503296),
        terms=(3, 3, 4),
        log_likelihood=-10.428077945053488,
        score=(-0.2258568869698424, -2.3064043981545264, 68.90781964636699),
        hessian=(
            0.022040360197258253,
            0.16132634783560174,
            0.20695057226041522,
            1.2540471973228067,
            11.606312676918371,
            -563.9377055455429,
        ),
        p_p_constituents=(-2.0258656646337854, -561.9119818703132, 0.0001419894039964265),
    ),
    FrozenTweedieCutoffCase(
        id="mid-left-adjacent-cutoff",
        row=(1.1, 0.9, 0.3, 1.49, 1.0),
        semantics="prior",
        cutoffs=(37.198668580305075, 37.19866858030508, 37.19866858030509),
        terms=(28, 28, 29),
        log_likelihood=-0.48619249762821254,
        score=(0.7799870673332028, -1.5282853694625647, -0.05346646226606078),
        hessian=(
            -5.19124725925098,
            -2.599956891110676,
            0.08217983962066192,
            3.964441942846876,
            -0.0025244737827308703,
            0.034055453298741156,
        ),
        p_p_constituents=(-112.58973375347755, 57.70749129424325, 54.91629791253304),
    ),
    FrozenTweedieCutoffCase(
        id="mid-right-adjacent-cutoff",
        row=(8.0, 2.0, 2.2, 1.51, 3.25),
        semantics="prior",
        cutoffs=(37.883368320677135, 37.88336832067714, 37.88336832067715),
        terms=(32, 33, 33),
        log_likelihood=-6.448153700850506,
        score=(3.1121221678285167, 1.6392073333724366, 3.6472897053085447),
        hessian=(
            -2.86833926468195,
            -1.4146009853765984,
            -2.1571587061884414,
            -1.6039343423246404,
            -2.1779228333859812,
            -5.953371116653781,
        ),
        p_p_constituents=(-202.50103671368535, 194.7179774301002, 1.829688166931366),
    ),
    FrozenTweedieCutoffCase(
        id="three-quarter-frequency-adjacent-cutoff",
        row=(3.4, 2.2, 1.1, 1.75, 3.0),
        semantics="frequency",
        cutoffs=(36.31621839739933, 36.31621839739934, 36.316218397399346),
        terms=(29, 30, 30),
        log_likelihood=-6.728986295387487,
        score=(0.8235123644386292, -1.2485287236195775, -1.6633379608797476),
        hessian=(
            -1.3413269572295858,
            -0.7486476040351174,
            -0.6493043850926206,
            0.5948518483797953,
            -0.4140961971623175,
            0.12789091652408047,
        ),
        p_p_constituents=(-392.5885429867951, 222.38232287662197, 170.33411102669723),
    ),
)


@dataclass(frozen=True, slots=True)
class FrozenTweedieCapBoundary:
    """A compiled/mirror endpoint boundary frozen before mirror deletion."""

    id: str
    row: tuple[float, float, float, float, float]
    semantics: WeightSemantics
    log_cutoff: float
    max_terms: int
    status: int
    failing_row: int
    terms: int
    log_likelihood: float
    deleted_mirror_outcome: str
    source: FrozenCharacterizationSource = "frozen-python-characterization/v1"


FROZEN_CUTOFF_CAP_BOUNDARIES = (
    FrozenTweedieCapBoundary(
        id="lower-positive-compiled-four-term-boundary",
        row=(1.8, 1.3, 0.6, 1.05, 0.4),
        semantics="prior",
        log_cutoff=38.58645794336696,
        max_terms=4,
        status=0,
        failing_row=-1,
        terms=4,
        log_likelihood=-1.5150198182932155,
        deleted_mirror_outcome="row 0: positive series window reached per-row max_terms=4",
    ),
    FrozenTweedieCapBoundary(
        id="tiny-positive-three-term-boundary",
        row=(0.015, 2.5, 1.4, 1.25, 0.4),
        semantics="prior",
        log_cutoff=38.03041909503295,
        max_terms=3,
        status=0,
        failing_row=-1,
        terms=3,
        log_likelihood=-10.428077945053488,
        deleted_mirror_outcome="success with 3 terms",
    ),
)


@dataclass(frozen=True, slots=True)
class FrozenTweedieRefusal:
    """Exact raw status and public refusal frozen before mirror deletion."""

    id: str
    row: tuple[float, float, float, float, float]
    semantics: WeightSemantics
    derivative_order: int
    max_terms: int
    log_cutoff: float
    status: int
    failing_row: int
    message: str
    source: FrozenCharacterizationSource = "frozen-python-characterization/v1"


FROZEN_TWEEDIE_REFUSALS = (
    FrozenTweedieRefusal(
        "per-row-work-limit",
        (1.1, 0.9, 0.3, 1.49, 1.0),
        "prior",
        0,
        2,
        37.0,
        4,
        0,
        "row 0: positive series window reached per-row max_terms=2",
    ),
    FrozenTweedieRefusal(
        "nonrepresentable-mode",
        (1.0, 1.0, 2.0e-16, 1.5, 1.0),
        "prior",
        0,
        100_000,
        37.0,
        2,
        0,
        "row 0: series mode lies above the exact float64 integer range 2**52",
    ),
    FrozenTweedieRefusal(
        "huge-cap-nonrepresentable-mode",
        (1.0, 1.0, 2.0e-16, 1.5, 1.0),
        "prior",
        0,
        2**100,
        37.0,
        2,
        0,
        "row 0: series mode lies above the exact float64 integer range 2**52",
    ),
    FrozenTweedieRefusal(
        "zero-rate-overflow",
        (0.0, 1.0e308, 1.0e-308, 1.5, 1.0),
        "prior",
        2,
        100_000,
        37.0,
        25,
        0,
        "row 0: required float64 work is not representable",
    ),
    FrozenTweedieRefusal(
        "zero-rate-underflow",
        (0.0, 1.0e-308, 1.0e308, 1.5, 1.0),
        "prior",
        2,
        100_000,
        37.0,
        16,
        0,
        "row 0: zero-atom rate is not representable",
    ),
    FrozenTweedieRefusal(
        "positive-series-base",
        (1.0, 1.0e308, 1.0e-308, 1.5, 1.0),
        "prior",
        2,
        100_000,
        37.0,
        18,
        0,
        "row 0: positive-row series base is not representable",
    ),
    FrozenTweedieRefusal(
        "positive-canonical-scale",
        (1.0e308, 1.0e-308, 1.0, 1.5, 1.0),
        "prior",
        2,
        100_000,
        37.0,
        17,
        0,
        "row 0: positive-row canonical scale is not representable",
    ),
    FrozenTweedieRefusal(
        "denominator-underflow-zero",
        (0.0, 1.0, 1.0e-200, 1.5, 1.0e-200),
        "prior",
        2,
        100_000,
        37.0,
        25,
        0,
        "row 0: required float64 work is not representable",
    ),
    FrozenTweedieRefusal(
        "denominator-underflow-positive",
        (1.0, 1.0, 1.0e-200, 1.5, 1.0e-200),
        "prior",
        2,
        100_000,
        37.0,
        25,
        0,
        "row 0: required float64 work is not representable",
    ),
    FrozenTweedieRefusal(
        "complete-value",
        (0.0, 1.0, 2.0e-300, 1.5, float(2**53)),
        "frequency",
        0,
        100_000,
        37.0,
        22,
        -1,
        "complete row value is not representable",
    ),
    FrozenTweedieRefusal(
        "complete-score",
        (0.0, 1.0, 1.0e-308, 1.5, 1.0e-308),
        "prior",
        1,
        100_000,
        37.0,
        23,
        -1,
        "complete row score is not representable",
    ),
    FrozenTweedieRefusal(
        "complete-Hessian",
        (0.0, 1.0, 1.0e-160, 1.5, 1.0e-160),
        "prior",
        2,
        100_000,
        37.0,
        24,
        -1,
        "complete row Hessian is not representable",
    ),
)
