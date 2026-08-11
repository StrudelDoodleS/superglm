"""Full-frame level binding, shared by ``cross_validate`` and ``bind_levels``.

Neutral home so the public one-call form on :class:`~superglm.SuperGLM` and the
``cross_validate`` pre-pass run the identical resolution without importing each
other (spec 2026-08-11, §9).
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm._frame import EagerFrame

logger = logging.getLogger(__name__)


def validate_binding_weight(sample_weight, n_rows: int) -> NDArray[np.floating] | None:
    """Validate a weight handed to the binding pre-pass, before anything reads it.

    The binding pass uses the weight to decide two things that persist into
    every later fit: which level is ``most_exposed`` enough to be the pinned
    base, and which declared levels have no EFFECTIVE rows and are therefore
    pinned. A silently wrong weight -- ragged length, NaN, a negative entry --
    does not raise there; it quietly moves the reference level or invents a pin,
    and the model then carries that decision through every fold.

    The checks are the ones ``cross_validate`` already applies to its own
    weight, minus the family-specific rule (Tweedie's strictly-positive
    requirement), which belongs to the fit that knows the family: binding only
    needs the weight to be a real, finite, nonnegative measure of exposure.
    """
    if sample_weight is None:
        return None
    try:
        raw = np.asarray(sample_weight)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("sample_weight must be a numeric one-dimensional array") from exc
    if raw.ndim != 1:
        raise ValueError("sample_weight must be one-dimensional")
    if len(raw) != n_rows:
        raise ValueError(f"sample_weight length {len(raw)} != X row count {n_rows}")
    if np.iscomplexobj(raw) or getattr(raw.dtype, "kind", None) in {"M", "m"}:
        raise ValueError("sample_weight must contain only real numeric values")
    try:
        weight = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("sample_weight must contain only real numeric values") from exc
    if not np.all(np.isfinite(weight)):
        raise ValueError("sample_weight must contain only finite values")
    if np.any(weight < 0.0):
        raise ValueError("sample_weight must be nonnegative")
    return weight


def auto_detected_templates(
    model,
    frame: EagerFrame,
    sample_weight,
    *,
    strict: bool = False,
) -> list[tuple[Any, Any]]:
    """Return the specs the fit path's own auto-detection builds on *frame*.

    Classification is not reimplemented here: a throwaway clone runs the very
    detection each fold will run, so a column that becomes categorical for the
    fold becomes categorical for the binding pass too.
    """
    if getattr(model._config, "splines", None) is None:
        # Without the splines shorthand an empty feature set is a configuration
        # error the fold fit reports; there is nothing to detect or bind.
        return []

    from superglm.model.base import auto_detect

    probe = model.clone_unfitted()
    try:
        auto_detect(probe, frame, sample_weight)
    except Exception as exc:
        # Detection failures belong to the fold that raises them, where
        # error_score decides the outcome; binding must not preempt that.
        if strict:
            raise
        logger.debug(f"Level binding skipped: feature auto-detection failed: {exc!r}")
        return []
    return [(name, probe._specs[name]) for name in probe._feature_order]


def resolve_level_bindings(
    model,
    frame: EagerFrame,
    sample_weight,
    *,
    strict: bool = False,
) -> dict[Any, Any]:
    """Bind level universes and most-exposed bases on the full pre-split frame.

    Sharing the level SET across folds is R factor semantics: the vocabulary is
    a property of the data column, not of the training subset, so no target
    information crosses folds (spec 2026-08-11, §3.5).  Quantities that do
    depend on training rows -- knots, penalties, coefficients -- keep binding
    per fold.

    ``strict`` marks the caller as owning the frame: ``bind_levels`` was handed
    THE universe source, so a column it cannot bind is the caller's error and
    is raised here.  The ``cross_validate`` pre-pass owns no such promise --
    the frame is the fold loop's input -- so it stays lenient and lets each
    fold report under the caller's ``error_score``.
    """
    config = getattr(model, "_config", None)
    if config is None:
        return {}
    templates: list[tuple[Any, Any]] = list(config.feature_templates)
    if not templates:
        templates = auto_detected_templates(model, frame, sample_weight, strict=strict)

    # Terms that declare their own universe (OrderedCategorical) or hold no
    # universe at all (numeric, spline) never grow the hook.
    bindable = [(name, spec) for name, spec in templates if hasattr(spec, "resolve_binding")]
    if strict:
        frame.require_columns(tuple(name for name, _ in bindable))

    available = set(frame.columns)
    bindings: dict[Any, Any] = {}
    for name, spec in bindable:
        if name not in available:
            # Only reachable when strict=False: strict mode already raised via
            # require_columns above. The CV pre-pass tolerates absent columns
            # so error_score still owns fold-level failures.
            continue
        probe = copy.deepcopy(spec)
        declared = frame.column_declared_categories(name)
        if declared is not None and hasattr(probe, "adopt_dtype_categories"):
            probe.adopt_dtype_categories(declared)
        try:
            bindings[name] = probe.resolve_binding(frame.column_array(name), sample_weight)
        except Exception as exc:
            # A column the whole frame cannot bind (missing values, data outside
            # a declared universe) is a fit error, and stays one: it is reported
            # per fold under the caller's error_score rather than raised here.
            if strict:
                raise
            logger.debug(f"Level binding skipped for feature {name!r}: {exc!r}")
    return bindings
