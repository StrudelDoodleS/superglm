"""Adapt internal numerical fixtures to the public family-bound constructor.

These fixtures intentionally retain Predictor's lower-level controls. Public
constructor tests should call SuperLSS and family helpers directly.
"""

from dataclasses import replace

from superglm import SuperLSS
from superglm.distributional.binding import _bind_predictor_template


def model_from_templates(*, family, predictors, **options) -> SuperLSS:
    return SuperLSS(
        family,
        *(_bind_predictor_template(family, template) for template in predictors),
        **options,
    )


def with_offset_declarations(model: SuperLSS, offsets) -> SuperLSS:
    """Attach numerical fixtures' known sources before calling the public fit.

    This helper changes only predictor declarations, never fit signatures or
    prediction defaults. Public declaration tests call family helpers directly.
    """
    if offsets is not None:
        names = {p.name for p in model._predictors}
        if set(offsets) - names:
            raise ValueError("unknown offset predictor name")
        model._predictors = tuple(replace(p, offset=offsets.get(p.name)) for p in model._predictors)
    return model
