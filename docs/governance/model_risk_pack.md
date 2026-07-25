# SuperGLM Model Risk Telemetry Pack

This document describes how to collect SuperGLM training telemetry for model
risk, software governance, and pricing model review. SuperGLM is tracking-system
agnostic: it does not import MLflow and does not expose a `superglm.mlflow`
integration. Instead, fitted models expose plain Python telemetry that project
orchestration code can log to MLflow, files, model cards, or another governance
system.

## Reproducibility Controls

A governed model run should identify the modelling code, fit controls, data
snapshots, and feature specification used to produce the fitted coefficients.
SuperGLM telemetry includes:

- SuperGLM package version.
- Model family, link, penalty settings, direct-solve mode, discrete setting, and
  binning metadata.
- Fit method, convergence mode, constructor tolerance, maximum iteration setting,
  final deviance, effective degrees of freedom, dispersion, and convergence.
- Ordered feature list, ordered interaction list, feature classes, interaction
  classes, and fit-time constraints.
- REML smoothing parameters, lambda history, optional objective history,
  optional inner iteration history, and REML profile fields when available.
- EDF by solver group and by feature term.

Project code should add run-level context that SuperGLM cannot know, such as:

- Source git SHA and package lockfile identifier.
- Training and validation data snapshot IDs.
- Feature-engineering pipeline version.
- Random seed and execution image.
- Pull request, model change ticket, and approval stage.

## Telemetry API

For a fitted model:

```python
model.fit_reml(X_train, y_train, sample_weight=w_train, offset=offset_train)

telemetry = model.training_telemetry()
reml = model.reml_diagnostics()
```

`reml["lambda_history"]` is the reliable REML path history when REML was used.
`reml["termination_reason"]` distinguishes score/objective convergence, an
active-set stationary point, fixed smoothing parameters, and iteration exhaustion.
`objective_history` and `inner_iter_history` are optional and path-dependent:
they are exposed when the underlying REML optimiser collected them, but callers
should not assume they are populated for every REML backend.

For scalar experiment-tracking metrics, use the dependency-free helper:

```python
from superglm.model.telemetry_ops import metrics_for_logging

scalar_metrics = metrics_for_logging(model, prefix="train")
```

For per-IRLS diagnostics, fit with diagnostics enabled:

```python
model.fit(X_train, y_train, record_diagnostics=True)
iteration_df = model.iteration_diagnostics()
```

For regularization paths:

```python
path = model.fit_path(X_train, y_train, sample_weight=w_train)
path_df = path.to_frame()
path_payload = path.to_telemetry()
```

These APIs return plain dictionaries or pandas DataFrames. They do not know
where the caller will log the information.

## Caller-Owned MLflow Example

If the training job already uses MLflow, pass the telemetry into MLflow from the
project layer:

```python
import json

from superglm.model.telemetry_ops import metrics_for_logging


def log_superglm_to_mlflow(mlflow, model, *, prefix="train"):
    telemetry = model.training_telemetry()

    mlflow.log_metrics(metrics_for_logging(model, prefix=prefix))
    mlflow.log_dict(telemetry, f"{prefix}_training_telemetry.json")

    try:
        iteration_df = model.iteration_diagnostics()
    except RuntimeError:
        iteration_df = None

    if iteration_df is not None:
        for row in iteration_df.to_dict("records"):
            step = int(row["iter"])
            mlflow.log_metrics(
                {
                    f"{prefix}.irls.deviance": row["deviance"],
                    f"{prefix}.irls.W_ratio": row["W_ratio"],
                    f"{prefix}.irls.step_halvings": row["step_halvings"],
                },
                step=step,
            )
        mlflow.log_text(
            iteration_df.to_json(orient="records"),
            f"{prefix}_iteration_diagnostics.json",
        )

    # Optional: log a stable copy through systems that do not support nested
    # dictionaries natively.
    mlflow.log_text(
        json.dumps(telemetry["features"], indent=2, sort_keys=True),
        f"{prefix}_feature_schema.json",
    )
```

Candidate interaction search should use parent and child runs in caller code:

```python
with mlflow.start_run(run_name="baseline-main-effects"):
    baseline.fit_reml(X_train, y_train, sample_weight=w_train)
    log_superglm_to_mlflow(mlflow, baseline, prefix="baseline")

    for candidate_name, candidate_model in candidates:
        with mlflow.start_run(run_name=candidate_name, nested=True):
            candidate_model.fit_reml(
                X_train,
                y_train,
                sample_weight=w_train,
                interaction_mode="fast_candidate",
            )
            log_superglm_to_mlflow(mlflow, candidate_model, prefix="candidate")
            mlflow.log_param("candidate.name", candidate_name)
            mlflow.log_metric("candidate.validation_gini", validation_gini)
            mlflow.log_metric("candidate.delta_deviance", delta_deviance)
```

The important boundary is that MLflow is supplied by the caller. SuperGLM only
provides telemetry.

## Interpretability

The telemetry feature schema provides a reviewable inventory of:

- Main effects and feature-spec classes.
- Interactions and parent features.
- Fit-time constraints, such as monotonicity constraints.
- Solver groups and group sizes.

The EDF payload separates effective degrees of freedom by solver group and by
feature term. Reviewers should use this to check that model flexibility is
concentrated in intended rating factors and interactions, and that constrained
terms remain interpretable on the linear predictor scale.

## Validation And Benchmarking

SuperGLM telemetry records training fit metrics. Validation metrics remain the
caller's responsibility because SuperGLM does not own the validation split or
business scoring policy.

For model-risk review, log validation artifacts from the project layer:

- Out-of-sample deviance and Gini.
- Calibration by relevant portfolio segments.
- Lift or decile charts for the target decision.
- Actual-to-expected ratios.
- Champion-challenger deltas.
- Runtime and fit-quality comparison for candidate versus final fitting modes.

## Limitations

Telemetry is not model approval. In particular:

- It does not store training or validation data.
- It does not calculate validation Gini, lift, fairness, or calibration metrics
  unless project code does so separately.
- It does not prove external feature engineering is reproducible.
- It does not certify that a monotone main effect remains globally monotone after
  adding unconstrained interactions involving the same variable.
- It does not replace independent code review, statistical review, or model risk
  approval.

If a model was fit with approximate or candidate-mode settings, label the run as
candidate telemetry and run a full final fit before production approval.

## Change Control

Recommended controls:

- Log source git SHA, package version, and dependency lockfile ID.
- Link the run to a pull request, model change ticket, and approval record.
- Keep candidate interaction runs nested under a parent search run.
- Record fit controls, data snapshot IDs, and validation split identifiers.
- Preserve telemetry, validation reports, and benchmark outputs before promoting
  a model to staging or production.

The goal is reproducibility and reviewability without coupling SuperGLM to any
single experiment-tracking backend.
