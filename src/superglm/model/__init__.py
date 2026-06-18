"""SuperGLM model package."""

from superglm.model.api import SuperGLM
from superglm.model.fit_ops import PathResult
from superglm.offsets import install_deployable_offset_support

install_deployable_offset_support(SuperGLM)
SuperGLM.__module__ = "superglm.model"

__all__ = ["SuperGLM", "PathResult"]
