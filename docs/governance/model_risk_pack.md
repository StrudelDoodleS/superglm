# SuperGLM Model Risk Pack

This document describes how to use SuperGLM's optional MLflow logging helper to
produce an auditable model run for model risk, software governance, and pricing
model review. The helper records metadata from an already-fitted model. It does
not refit the model, change predictions, or alter fitting behaviour.

## Reproducibility Controls

A governed SuperGLM run should identify the exact modelling code, fit settings,
and feature specification used to produce the fitted coefficients. The MLflow
helper records:

- SuperGLM package version and source git SHA when available.
- Model family, link, penalty settings, direct-solve mode, discrete setting, and
  binning metadata.
- Fit method, tolerances, maximum iteration settings, convergence mode, and
  REML profile fields when available.
- Ordered feature list, ordered interaction list, feature classes, interaction
  classes, and constraints.
- Smoothing parameters in `lambdas.json`.
- EDF attribution in `edf_by_term.json`.

For production review, also record the training data snapshot identifier,
validation data snapshot identifier, random seed, environment image, and any
external rating-table or exposure filters as user-supplied MLflow tags or
parameters.

## Interpretability

SuperGLM models are designed to keep the modelling surface inspectable. The
logged feature schema provides a reviewable inventory of:

- Main effects and their feature-spec classes.
- Interactions and their parent features.
- Fit-time constraints, such as monotonicity constraints.
- Solver groups and group sizes.

The EDF artifact separates effective degrees of freedom by solver group and by
feature term. Reviewers should use this to check that model flexibility is
concentrated in intended rating factors and interaction terms, and that
constrained terms remain interpretable on the linear predictor scale.

## Validation And Benchmarking

The MLflow helper logs model-fit metrics directly from the fitted model when
available, including deviance, effective degrees of freedom, convergence flags,
IRLS iterations, REML iterations, and fit statistics such as log likelihood or
explained deviance.

Validation metrics are supplied by the caller because the helper does not own
the validation split or business scoring policy. Pass out-of-sample Gini,
calibration, lift, actual-to-expected ratios, benchmark comparisons, and
champion-challenger deltas through `validation_metrics`. Numeric values are
logged as MLflow metrics under `validation.*`, and the complete nested payload is
logged to `validation_metrics.json`.

For material model changes, the governance pack should include:

- Train and validation performance on fixed data snapshots.
- Calibration by relevant portfolio segments.
- Lift or decile charts for the target business decision.
- Stability checks versus the previous production model.
- Runtime and fit-quality comparison for any alternative fitting mode, such as
  candidate interaction screening versus full REML.

## Limitations

The MLflow helper records what the fitted model exposes. It does not guarantee
that every possible business validation has been run. In particular:

- It does not store training data or validation data by default.
- It does not calculate Gini, calibration, lift, or fairness metrics unless the
  caller supplies them.
- It does not prove that feature engineering outside SuperGLM is reproducible.
- It does not certify that constraints remain valid after adding unconstrained
  interactions involving the same variables.
- It does not replace independent code review, statistical review, or model risk
  approval.

If the fitted model was produced with approximate or candidate-mode settings,
label the run clearly using tags such as `run_type=candidate_interaction` and do
not treat it as the final governed production fit without a full final run.

## Change Control

Each governed model change should have a traceable source-control and review
path. Recommended controls:

- Log the git SHA and package version for every run.
- Link the MLflow run to the pull request, issue, model change ticket, or
  approval record using tags.
- Keep candidate search runs nested under a parent experiment run so reviewers
  can distinguish explored interactions from the selected model.
- Record fit controls and validation data identifiers as tags or parameters.
- Preserve model artifacts and validation reports in the MLflow run before
  promoting a model to staging or production.

Use nested runs for repeated candidate interaction searches:

```python
import superglm.mlflow as superglm_mlflow

parent_run_id = superglm_mlflow.log_model_run(
    baseline_model,
    experiment_name="pricing-governance",
    run_name="baseline-main-effects",
    run_type="baseline",
    validation_metrics=baseline_metrics,
)

candidate_run_id = superglm_mlflow.log_model_run(
    candidate_model,
    experiment_name="pricing-governance",
    run_name="candidate-age-area",
    nested=True,
    run_type="candidate_interaction",
    validation_metrics=candidate_metrics,
    tags={"parent_run_id": parent_run_id},
)
```

## What MLflow Captures

`superglm.mlflow.log_model_run()` logs five JSON artifacts:

- `model_config.json`: package version, git SHA, family, link, penalty settings,
  discrete settings, fit tolerances, convergence status, REML metadata, and fit
  statistics.
- `feature_schema.json`: features, interactions, constraints, and solver group
  metadata.
- `lambdas.json`: smoothing parameters by term or component.
- `edf_by_term.json`: effective degrees of freedom by solver group, by feature,
  and in total.
- `validation_metrics.json`: caller-supplied validation, lift, calibration, or
  benchmark metrics.

It also logs scalar MLflow metrics for deviance, effective degrees of freedom,
convergence, IRLS/REML iterations, smoothing parameters, EDF by group, and any
numeric validation metrics. Parameters and tags summarize model configuration,
feature counts, interaction counts, constraint counts, version, run type, and
git SHA.

MLflow is an optional dependency. Importing `superglm.mlflow` does not require
MLflow to be installed. Calling `log_model_run()` without MLflow installed raises
a clear installation error.
