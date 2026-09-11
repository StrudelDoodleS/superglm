import pytest

from superglm.distributional.families.gamma import GammaLS
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.families.generalized_gamma import GeneralizedGammaLSS
from superglm.distributional.families.generalized_pareto import GeneralizedParetoLSS
from superglm.distributional.families.log_normal import LogNormalLS
from superglm.distributional.families.negative_binomial import NegativeBinomialLS
from superglm.distributional.families.tweedie import TweedieLSS
from superglm.distributional.families.two_piece import TwoPieceLogNormalLSS, TwoPieceNormalLSS


def test_tweedie_helpers_bind_actual_configured_family():
    from superglm.distributional.families.tweedie import TweedieLSS

    family = TweedieLSS(power_lower=1.08, power_upper=1.92)
    assert callable(getattr(family, "mu", None)), "family helpers are not implemented"
    assert family.mu("x").name == "mean"
    assert family.phi().name == "dispersion"
    assert family.p().family is family


def _resolve(family, predictors):
    from superglm.distributional import binding

    assert callable(getattr(binding, "resolve_predictors", None)), "resolver is not implemented"
    return binding.resolve_predictors(family, predictors)


@pytest.mark.parametrize(
    "family,helpers,names",
    [
        (GaussianLS(), ("location", "scale"), ("location", "scale")),
        (GammaLS(), ("mean", "scale"), ("mean", "scale")),
        (TweedieLSS(), ("mu", "phi", "p"), ("mean", "dispersion", "power")),
        (NegativeBinomialLS(), ("mean", "theta"), ("mean", "theta")),
        (GeneralizedParetoLSS(), ("scale", "shape"), ("scale", "shape")),
        (GeneralizedGammaLSS(), ("mean", "scale", "shape"), ("mean", "scale", "shape")),
        (
            GeneralizedGammaLSS(parametrisation="location"),
            ("location", "scale", "shape"),
            ("location", "scale", "shape"),
        ),
        (LogNormalLS(), ("mean", "scale"), ("mean", "scale")),
        (LogNormalLS(parametrisation="location"), ("location", "scale"), ("location", "scale")),
        (TwoPieceLogNormalLSS(), ("mean", "scale", "skew"), ("mean", "scale", "skew")),
        (
            TwoPieceLogNormalLSS(parametrisation="location"),
            ("location", "scale", "skew"),
            ("location", "scale", "skew"),
        ),
        (TwoPieceNormalLSS(), ("location", "scale", "skew"), ("location", "scale", "skew")),
    ],
)
def test_every_family_resolves_helpers_in_canonical_order(family, helpers, names):
    assert all(callable(getattr(family, name, None)) for name in helpers)
    declarations = [getattr(family, helper)("x") for helper in reversed(helpers)]
    owned, templates = _resolve(family, declarations)
    assert owned is not family
    assert tuple(template.name for template in templates) == names
    assert tuple(parameter.name for parameter in owned.parameters) == names
    assert all(tuple(template.features) == ("x",) for template in templates)


@pytest.mark.parametrize("family_type", [GeneralizedGammaLSS, LogNormalLS, TwoPieceLogNormalLSS])
@pytest.mark.parametrize("mode,invalid", [("mean", "location"), ("location", "mean")])
def test_mode_dependent_helpers_reject_inactive_parameter(family_type, mode, invalid):
    family = family_type(parametrisation=mode)
    assert callable(getattr(family, invalid, None))
    with pytest.raises(ValueError, match=mode):
        getattr(family, invalid)("x")


@pytest.mark.parametrize(
    "provided,missing",
    [
        (("mu", "phi"), ("power",)),
        (("mu",), ("dispersion", "power")),
        ((), ("mean", "dispersion", "power")),
    ],
)
def test_missing_predictors_show_each_required_helper(provided, missing):
    family = TweedieLSS()
    with pytest.raises(ValueError) as error:
        _resolve(family, [getattr(family, helper)() for helper in provided])
    message = str(error.value)
    assert "TweedieLSS" in message
    assert message.split("SuperLSS(", 1)[1].lstrip().startswith("family,")
    helpers = {"mean": "mu", "dispersion": "phi", "power": "p"}
    for name in missing:
        assert name in message
    marked = [line for line in message.splitlines() if "<---" in line or "--->" in line]
    assert len(marked) == len(missing)
    assert all(any(f"family.{helpers[name]}(...)" in line for line in marked) for name in missing)
    assert "constant" not in message.lower()
    assert "intercept" not in message.lower()


def test_duplicate_predictors_are_rejected_before_missing_advice():
    family = TweedieLSS()
    with pytest.raises(ValueError, match="[Dd]uplicate.*mean") as error:
        _resolve(family, [family.mu(), family.mu()])
    assert "SuperLSS(" not in str(error.value)


@pytest.mark.parametrize("foreign", [TweedieLSS(), TweedieLSS(power_lower=1.08, power_upper=1.92)])
def test_equal_and_differently_configured_foreign_instances_are_rejected(foreign):
    family = TweedieLSS()
    with pytest.raises(ValueError, match="family instance") as error:
        _resolve(family, [family.mu(), foreign.phi()])
    assert "SuperLSS(" not in str(error.value)


def test_bare_helpers_receive_call_hint_before_completeness():
    family = TweedieLSS()
    with pytest.raises(TypeError, match=r"family\.phi\(.*\)") as error:
        _resolve(family, [family.mu(), family.phi])
    assert "call" in str(error.value).lower()
    assert "SuperLSS(" not in str(error.value)


@pytest.mark.parametrize("value", ["x", 3, None, {}, lambda: None])
def test_invalid_positional_values_do_not_render_object_repr(value):
    family = TweedieLSS()
    with pytest.raises(TypeError, match="BoundPredictor") as error:
        _resolve(family, [value])
    assert "0x" not in str(error.value)
    assert "SuperLSS(" not in str(error.value)


@pytest.mark.parametrize("value", ["x", b"x", {}, None])
def test_resolver_requires_a_sequence(value):
    with pytest.raises(TypeError, match="sequence"):
        _resolve(TweedieLSS(), value)


def test_binding_preserves_controls_interactions_and_owned_mutable_terms():
    from superglm.features import Spline
    from superglm.links import LogLink
    from superglm.terms import s, term, ti

    family = TweedieLSS()
    original = Spline(n_knots=5)
    terms = [ti("x", "z"), term("x", original), s("z", n_knots=6)]
    declaration = family.mu(*terms, intercept=False, link=LogLink())
    original.n_knots = 20
    terms.clear()
    template = declaration.template
    assert template.features["x"].n_knots == 5
    assert template.features["z"].n_knots == 6
    assert template.interaction_order == ("x:z",)
    assert not template.intercept
    assert isinstance(template.link, LogLink)
    template.features["x"].n_knots = 30
    assert declaration.template.features["x"].n_knots == 5
    assert declaration.template.link is not template.link
    _, first = _resolve(family, [declaration, family.phi(), family.p()])
    first[0].features["x"].n_knots = 40
    _, second = _resolve(family, [declaration, family.phi(), family.p()])
    assert second[0].features["x"].n_knots == 5


def test_private_template_binding_retains_pending_and_explicit_interactions():
    from superglm.distributional import binding
    from superglm.distributional.predictor import Predictor
    from superglm.features import Numeric
    from superglm.features.interaction import NumericInteraction

    assert callable(getattr(binding, "_bind_predictor_template", None))
    family = TweedieLSS()
    original = Predictor(
        "mean",
        {"x": Numeric(), "z": Numeric()},
        intercept=False,
        link="log",
        interactions=(("x", "z"),),
        interaction_specs={"product": NumericInteraction("x", "z")},
        interaction_order=("product",),
    )
    bound = binding._bind_predictor_template(family, original)
    template = bound.template
    assert template.interactions == (("x", "z"),)
    assert template.interaction_order == ("product",)
    assert template.interaction_specs["product"].parent_names == ("x", "z")
    assert template.interaction_specs["product"] is not original.interaction_specs["product"]
    assert template.link == "log"
    assert not template.intercept


class _FourParameterFamily:
    """Structural solver-family fixture with mutable, nested configuration."""

    def __init__(self):
        self.config = {"lower": [0.1]}

    @property
    def parameters(self):
        from superglm.distributional.families.gaussian import LowerBoundedLogLink
        from superglm.distributional.family import ParameterSpec, ParameterSupport

        return tuple(
            ParameterSpec(
                name,
                LowerBoundedLogLink(self.config["lower"][0]),
                name,
                ParameterSupport(lower=self.config["lower"][0]),
            )
            for name in ("a", "b", "c", "d")
        )

    def bind_likelihood(self, *args, **kwargs):
        raise NotImplementedError

    def initialize(self, *args, **kwargs):
        raise NotImplementedError

    def evaluate_natural(self, *args, **kwargs):
        raise NotImplementedError


def test_custom_family_snapshots_configuration_without_new_protocol_methods():
    from superglm.distributional.binding import bind_predictor

    family = _FourParameterFamily()
    declarations = [bind_predictor(family, name, "x") for name in ("d", "b", "a", "c")]
    owned, templates = _resolve(family, declarations)
    family.config["lower"][0] = 0.2
    assert owned.config["lower"] == [0.1]
    assert owned.parameters[0].support.lower == 0.1
    assert owned.parameters[0].default_link.floor == 0.1
    assert tuple(template.name for template in templates) == ("a", "b", "c", "d")
    assert all(declaration.family is family for declaration in declarations)
    other_owned, _ = _resolve(family, declarations)
    assert other_owned.config["lower"] == [0.2]


def test_custom_missing_diagnostic_uses_generic_binder():
    from superglm.distributional.binding import bind_predictor

    family = _FourParameterFamily()
    with pytest.raises(ValueError) as error:
        _resolve(family, [bind_predictor(family, name) for name in ("a", "b", "c")])
    message = str(error.value)
    marked = [line for line in message.splitlines() if "<---" in line]
    assert len(marked) == 1
    assert 'bind_predictor(family, "d", ...)' in marked[0]
    assert "family.d(" not in message


def test_generic_binder_rejects_unknown_names_and_invalid_controls():
    from superglm.distributional.binding import bind_predictor

    family = TweedieLSS()
    with pytest.raises(ValueError, match="valid predictors.*mean.*dispersion.*power"):
        bind_predictor(family, "mu")
    with pytest.raises(TypeError, match="intercept"):
        family.mu(intercept=1)
    with pytest.raises(TypeError, match="link"):
        family.mu(link=3)
    with pytest.raises(TypeError):
        family.mu(3)


@pytest.mark.parametrize("copy_behavior", ["raise", "return_self"])
def test_custom_family_that_cannot_be_snapshotted_fails_clearly(copy_behavior):
    from superglm.distributional.binding import bind_predictor

    class UnsnapshotableFamily(_FourParameterFamily):
        def __deepcopy__(self, memo):
            if copy_behavior == "raise":
                raise RuntimeError("custom copy is unavailable")
            return self

    family = UnsnapshotableFamily()
    declarations = [bind_predictor(family, name) for name in ("a", "b", "c", "d")]
    with pytest.raises(TypeError, match="family.*snapshot"):
        _resolve(family, declarations)
    foreign = UnsnapshotableFamily()
    with pytest.raises(ValueError, match="family instance"):
        _resolve(family, [bind_predictor(foreign, "a")])


def test_configured_tweedie_snapshot_retains_its_power_support_and_link():
    family = TweedieLSS(power_lower=1.08, power_upper=1.92)
    owned, templates = _resolve(family, [family.p(), family.phi(), family.mu()])
    assert (owned.power_lower, owned.power_upper) == (1.08, 1.92)
    parameter = owned.parameters[2]
    assert (parameter.support.lower, parameter.support.upper) == (1.08, 1.92)
    assert (parameter.default_link.lower, parameter.default_link.upper) == (1.08, 1.92)
    assert all(template.intercept and not template.features for template in templates)
