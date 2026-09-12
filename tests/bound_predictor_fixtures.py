"""Adapt internal numerical fixtures to the public family-bound constructor.

These fixtures intentionally retain Predictor's lower-level controls. Public
constructor tests should call SuperLSS and family helpers directly.
"""

from superglm import SuperLSS
from superglm.distributional.binding import _bind_predictor_template


def model_from_templates(*, family, predictors, **options) -> SuperLSS:
    return SuperLSS(
        family,
        *(_bind_predictor_template(family, template) for template in predictors),
        **options,
    )
