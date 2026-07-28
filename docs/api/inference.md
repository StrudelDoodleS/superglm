# Inference

## Term inference

::: superglm.TermInference

::: superglm.InteractionInference

::: superglm.SplineMetadata

## Credibility

::: superglm.RandomEffectResult

`FactorSmoothResult` is basis-aware. FS tables include local credibility,
shrinkage, and collapse diagnostics. SZ tables report symmetric raw-level EDF
and support without those labels; `collapsed` is `None`, and
`diagnostics["max_abs_level_effect_sum"]` audits the pointwise constraint.

::: superglm.FactorSmoothResult

## REML

::: superglm.REMLResult

## Profile estimation

::: superglm.NBProfileResult

::: superglm.TweedieProfileResult

::: superglm.estimate_nb_theta

::: superglm.estimate_tweedie_p

::: superglm.estimate_phi
