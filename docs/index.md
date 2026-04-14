<div class="sg-hero">
  <div class="sg-hero__logo">
    <img src="images/logo.png" alt="SuperGLM logo">
  </div>
  <div class="sg-hero__copy">
    <h1>SuperGLM</h1>
    <p>
      Penalised GLMs and GAM-style pricing models for insurance, with exact and
      discrete REML, solver-backed monotone splines, out-of-fold validation,
      and deployable fitted estimators.
    </p>
    <div class="sg-hero__actions">
      <a class="md-button md-button--primary" href="guide/workflows/">Recommended Workflows</a>
      <a class="md-button" href="guide/monotone/">Monotone Splines</a>
      <a class="md-button" href="api/model/">API Reference</a>
    </div>
  </div>
</div>

## Start Here

If you are new to the package, the intended path is:

1. build an explicit feature spec
2. fit with `fit_reml()` and `selection_penalty=0`
3. use `cross_validate(..., fit_mode="fit_reml")` for model comparison
4. validate challengers with Lorenz and double-lift charts
5. serialize the fitted estimator for scoring

## What It Covers

<div class="sg-feature-grid">
  <div class="sg-feature-card">
    <h3>Workflow-first pricing models</h3>
    <p>Exact REML for standard GAM-style pricing, discrete REML for large data, and clear guidance on when to use sparse selection instead.</p>
  </div>
  <div class="sg-feature-card">
    <h3>Feature engineering in-model</h3>
    <p>P-splines, cubic regression splines, natural splines, ordered categoricals, grouped categoricals, and interaction terms.</p>
  </div>
  <div class="sg-feature-card">
    <h3>Monotone solvers</h3>
    <p>Solver-backed monotone spline fitting through QP and SCOP paths, with post-fit repair retained only as a fallback.</p>
  </div>
  <div class="sg-feature-card">
    <h3>Validation and deployment</h3>
    <p>Cross-validation with out-of-fold predictions, Lorenz and double-lift charts, diagnostics, and deployable fitted estimators.</p>
  </div>
</div>

## Example Plots

Poisson frequency model on French MTPL2 (678k policies), fitted with REML
smoothness selection. Plots show pointwise confidence bands, weighted density,
and interior knot positions.

| Vehicle Age (`quantile_rows` knots) | Bonus-Malus (`quantile_tempered`, alpha=0.2) |
|:---:|:---:|
| [![VehAge](images/readme_vehage.png)](images/readme_vehage.png) | [![BonusMalus](images/readme_bonusmalus.png)](images/readme_bonusmalus.png) |
