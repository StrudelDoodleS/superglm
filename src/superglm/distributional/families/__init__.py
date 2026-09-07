"""Built-in distributional families."""

_EXPORTS = {
    "BoundedLogitLink": ("superglm.distributional.families._links", "BoundedLogitLink"),
    "GammaLS": ("superglm.distributional.families.gamma", "GammaLS"),
    "GaussianLS": ("superglm.distributional.families.gaussian", "GaussianLS"),
    "GeneralizedGammaLSS": (
        "superglm.distributional.families.generalized_gamma",
        "GeneralizedGammaLSS",
    ),
    "GeneralizedParetoLSS": (
        "superglm.distributional.families.generalized_pareto",
        "GeneralizedParetoLSS",
    ),
    "LogNormalLS": ("superglm.distributional.families.log_normal", "LogNormalLS"),
    "LowerBoundedLogLink": (
        "superglm.distributional.families.gaussian",
        "LowerBoundedLogLink",
    ),
    "NegativeBinomialLS": (
        "superglm.distributional.families.negative_binomial",
        "NegativeBinomialLS",
    ),
    "NegativeBinomialPoissonBoundaryError": (
        "superglm.distributional.families.negative_binomial",
        "NegativeBinomialPoissonBoundaryError",
    ),
    "TweedieLSS": ("superglm.distributional.families.tweedie", "TweedieLSS"),
    "TwoPieceLogNormalLSS": (
        "superglm.distributional.families.two_piece",
        "TwoPieceLogNormalLSS",
    ),
    "TwoPieceNormalLSS": (
        "superglm.distributional.families.two_piece",
        "TwoPieceNormalLSS",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    import importlib

    module_name, attribute_name = _EXPORTS[name]
    value = getattr(importlib.import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
