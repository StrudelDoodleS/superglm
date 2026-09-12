# SuperLSS

Pass a family first, followed by one predictor declaration for each parameter.
The [walkthrough](../getting-started/distributional.md) runs from sample data
through fitting, prediction and model comparison. The
[family reference](families.md#distributional-families) describes the helper
methods and parameter meanings; [term helpers](features.md#predictor-term-helpers)
describe the inputs to those methods.

## Model

::: superglm.SuperLSS
    options:
      members:
        - family
        - predictors
        - fit
        - fit_reml
        - parameter_names_
        - predict
        - predict_parameters
        - predict_link
        - predict_cdf
        - predict_quantile
        - summary
        - scores
        - compare
        - diagnose
        - to_bytes
        - from_bytes

## Predictor declarations

Built-in family helpers return a `BoundPredictor`. A custom family can use
`bind_predictor` with a name from its parameter definitions. Declarations
belong to the family instance that created them. Pass that same instance to
`SuperLSS`.

::: superglm.BoundPredictor
    options:
      members:
        - name
        - template

::: superglm.bind_predictor
