"""Relativity plotting for SuperGLM models."""

from superglm.plotting.comparison import plot_term_comparison
from superglm.plotting.interactions import plot_interaction
from superglm.plotting.main_effects import plot_relativities, plot_term
from superglm.plotting.main_effects_plotly import plot_main_effects_plotly

__all__ = [
    "plot_term_comparison",
    "plot_relativities",
    "plot_term",
    "plot_interaction",
    "plot_main_effects_plotly",
]
