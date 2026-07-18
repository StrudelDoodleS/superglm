"""Attempt-local model state for strongly exception-safe fitting."""

from __future__ import annotations

from dataclasses import dataclass

_ATTEMPT_RUNTIME_OPTIONS = ("_max_analytical_per_w", "_select_snap")


@dataclass
class FitWorkspace:
    """Fresh mutable state for one fit attempt."""

    model: object
    mode: str
    validated_inputs: object | None
    previous_revision: int

    @classmethod
    def start(
        cls,
        public_model,
        *,
        mode: str,
        validated_inputs,
        config_overrides: dict[str, object] | None = None,
    ):
        """Materialize only configuration intent, never prior fitted buffers."""
        config = public_model._config
        if config_overrides:
            config = config.with_value(**config_overrides)
        work_model = config.materialize(type(public_model))
        for name in _ATTEMPT_RUNTIME_OPTIONS:
            if hasattr(public_model, name):
                setattr(work_model, name, getattr(public_model, name))
        return cls(
            model=work_model,
            mode=mode,
            validated_inputs=validated_inputs,
            previous_revision=int(public_model._fit_revision),
        )
