"""SuperGLM model package."""

from superglm.model.api import SuperGLM
from superglm.model.fit_ops import PathResult

SuperGLM.__module__ = "superglm.model"

__all__ = ["SuperGLM", "PathResult"]
