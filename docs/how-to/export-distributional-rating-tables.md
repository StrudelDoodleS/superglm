# Export rating tables for a distributional model

A fitted {py:class}`~superglm.SuperLSS` can be written out as rating tables, as a
{py:class}`~superglm.SuperGLM` can (see [Deployment](deploy-a-fitted-model.md)). There is one
set of tables for each parameter that is a product of factors, so a rater can price the mean and
the variability of a risk from the same sheet of lookups.

```python
payload = model.rating_table_payload(train, offsets={"mean": np.log(exposure)})
model.export_rating_tables("tables.xlsx", train, offsets={"mean": np.log(exposure)})
```

`train` is the book the tables describe: it gives each level its exposure, and the columns any
offset is read from. `offsets` is what `predict_parameters` takes, and is needed for the same
offsets. Pass `sample_weight=` to weight the exposure column.

## What each table multiplies

Each parameter's table is a base and one factor per term, interaction and offset. Multiply them
for a row and the `rule` says what you hold:

| Family and parameter | The table is | The model's parameter is |
| --- | --- | --- |
| any `mean` on a log link | Mean | the product |
| Gamma `scale` | CV (coefficient of variation) | the product |
| log-normal `scale` | Sigma above its floor | floor + product |
| negative-binomial `theta` | Overdispersion (1/θ) | 1 / product |

The floor of a log-normal sigma is written as its own cell, `Floor`, on the sheet. Leaving it out
misprices every row by the floor's share of sigma: with the default floor of 0.01,
2% at a sigma of 0.5 and 10% at a sigma of 0.1. The sheet also gives CV = sqrt(exp(σ²) − 1).
For the negative binomial a higher factor always means more variance:
variance = mean + mean² × overdispersion, and an offset enters as exp(−offset).

The **Summary** sheet lists each table under the editor's name for its parameter (Mean, CV,
Overdispersion, Dispersion), with the name used in code beside it. Its **Floor** column appears only
when some parameter has a floor, which today means a log-normal. Its notes define **Weight**, the sum
of the sample weights in a bin or level (the row count when no weights were given), and for a
Tweedie model give the chance of no claim: exp(−λ) with λ = mean^(2 − p) / (dispersion × (2 − p)),
where p is the model's power.

## What is left out

A parameter on any other link, such as the identity-link location of a Gaussian, has no table,
because a term changes it by adding to it. A parameter holding a term that has no exact table is
also left out: `Polynomial` and `Piecewise` terms, surface interactions (`ti`),
spline-by-categorical interactions, and grouped categoricals. Each left-out parameter gets one
sentence in `payload.omitted`, and the others are still exported. If no parameter qualifies,
the call raises `ValueError` with every sentence.

## Reading the payload

`payload.parameters` holds one table set per parameter, with `quantity`, `link`, `rule`, `floor`,
`base_relativity`, `main_effects` and `interactions`. `payload.blocks()` keys every block as
`"<parameter>:<term>"`, such as `"mean:age"`. A spline is its exact piecewise cubic, read
as for a SuperGLM workbook in [Deployment](deploy-a-fitted-model.md); a categorical is one factor per level against its reference level. The tables follow the
coefficients the model predicts with, so a model whose coefficients were edited by hand is
tabulated as edited.
