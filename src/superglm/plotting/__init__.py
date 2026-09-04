"""Relativity and distributional plotting for SuperGLM and SuperLSS models."""

from superglm.plotting.comparison import plot_term_comparison
from superglm.plotting.distributional import (
    plot_actual_expected,
    plot_binned,
    plot_binned_2d,
    plot_calibration,
    plot_comparison,
    plot_density_fan,
    plot_diagnostics_figure,
    plot_pit,
    plot_portfolio,
    plot_qq,
    plot_risk_curves,
    plot_spread,
    plot_term_effect,
    plot_term_grid,
    plot_worm,
)
from superglm.plotting.interactions import plot_interaction
from superglm.plotting.main_effects import plot_relativities, plot_term
from superglm.plotting.main_effects_plotly import plot_main_effects_plotly

# The plotly renderers live in a module that cannot be imported without plotly,
# so they are resolved on first use exactly as the scalar explorer resolves its
# own plotly import: importing this package needs matplotlib only.  They stay
# out of ``__all__`` because plotly is the optional ``plotting`` extra and every
# name in ``__all__`` must resolve under the base install; ``dir()`` lists them.
_PLOTLY_EXPORTS = (
    "plotly_actual_expected",
    "plotly_binned",
    "plotly_binned_2d",
    "plotly_calibration",
    "plotly_comparison",
    "plotly_density_fan",
    "plotly_diagnostics_figure",
    "plotly_pit",
    "plotly_portfolio",
    "plotly_qq",
    "plotly_risk_curves",
    "plotly_spread",
    "plotly_term_effect",
    "plotly_term_grid",
    "plotly_worm",
)

__all__ = [
    "plot_actual_expected",
    "plot_binned",
    "plot_binned_2d",
    "plot_calibration",
    "plot_comparison",
    "plot_density_fan",
    "plot_diagnostics_figure",
    "plot_interaction",
    "plot_main_effects_plotly",
    "plot_pit",
    "plot_portfolio",
    "plot_qq",
    "plot_relativities",
    "plot_risk_curves",
    "plot_spread",
    "plot_term",
    "plot_term_comparison",
    "plot_term_effect",
    "plot_term_grid",
    "plot_worm",
]


def __getattr__(name: str):
    if name not in _PLOTLY_EXPORTS:
        raise AttributeError(name)
    import importlib

    value = getattr(importlib.import_module("superglm.plotting.distributional_plotly"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_PLOTLY_EXPORTS))
