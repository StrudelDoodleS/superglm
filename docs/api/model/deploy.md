---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Deploy

```{code-cell} ipython3
:tags: [remove-cell]

import logging
from pathlib import Path

import matplotlib.pyplot as plt

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
for candidate in (
    Path("../../_static/superglm.mplstyle"),
    Path("docs/_static/superglm.mplstyle"),
):
    if candidate.exists():
        plt.style.use(str(candidate))
        break
else:
    raise FileNotFoundError("superglm.mplstyle: run this page from its own directory")
```

{py:meth}`~superglm.SuperGLM.export_rating_tables` writes the deployment
rating tables for the fitted model;
{py:meth}`~superglm.SuperGLM.rating_table_payload` builds the
renderer-independent payload behind them, for when you need the tables as
objects rather than files. The [how-to on deploying a fitted
model](../../how-to/deploy-a-fitted-model.md) shows the export end to end.

```{eval-rst}
.. autosummary::
   :toctree: ../generated
   :nosignatures:

   ~superglm.SuperGLM.export_rating_tables
   ~superglm.SuperGLM.rating_table_payload
```

## Example

The same simulated motor book as the rest of this section: 4,000 policies, a
non-linear age effect, four regions and a mild vehicle-power slope, with
exposure carried as a log offset. Export needs the fitting frame back, because
the tables carry the weight behind every band.

```{code-cell} ipython3
import numpy as np
import pandas as pd

from superglm import Categorical, Numeric, Spline, SuperGLM

rng = np.random.default_rng(0)
n = 4000
region = rng.choice(["North", "South", "East", "West"], n, p=[0.35, 0.3, 0.2, 0.15])
age = rng.uniform(18, 80, n)
veh_power = rng.uniform(4, 12, n)
exposure = rng.uniform(0.1, 1.0, n)
region_effect = pd.Series(region).map(
    {"North": 0.0, "South": 0.25, "East": -0.2, "West": 0.4}
).to_numpy()
log_rate = (
    -1.2
    + 1.1 * np.exp(-((age - 24) ** 2) / 90.0)
    + 0.004 * (age - 50) ** 2 / 10.0
    + region_effect
    + 0.09 * (veh_power - 8)
)
claims = rng.poisson(np.exp(log_rate) * exposure)
X = pd.DataFrame({"age": age, "region": region, "veh_power": veh_power})
offset = np.log(exposure)

model = SuperGLM(
    family="poisson",
    features={
        "age": Spline(kind="cr", k=10),
        "region": Categorical(),
        "veh_power": Numeric(),
    },
).fit_reml(X, claims, offset=offset)
model.reml_diagnostics()["converged"]
```

`rating_table_payload` is the objects behind the workbook. Each main effect
becomes a block that knows its own shape: the age smooth is banded into a
continuous grid, region stays categorical, the numeric power term exports as a
single per-unit factor, and the log offset becomes its own multiplier block.
The base relativity is the rest of the rate — what a risk pays before any
factor applies.

```{code-cell} ipython3
payload = model.rating_table_payload(X, claims, offset=offset)
print(
    f"base relativity {payload.base_relativity:.4f}, "
    f"{payload.selected_n_bins} bands chosen for the banded curves"
)
pd.DataFrame(
    [
        {"block": block.name, "kind": block.kind, "rows": len(block.table)}
        for block in payload.main_effects
    ]
)
```

A block's `table` is an ordinary DataFrame, keyed by the rating value with the
relativity and the fitting weight behind it — one per row here, because no
`sample_weight` was passed. This is the table a rater keys on.

```{code-cell} ipython3
region_block = next(
    block for block in payload.main_effects if block.name == "region"
)
region_block.table.round(3)
```

Banding a smooth curve is an approximation, and the payload measures it rather
than asserting it: `discretization_impact` refits at each candidate band count
and reports what the banded model does to the deviance and to individual
predictions. The exported grid changes the deviance by about a hundredth of a
percent and no single prediction by more than two percent.

```{code-cell} ipython3
payload.discretization_impact[
    [
        "feature",
        "n_bins",
        "exported",
        "deviance_change_pct",
        "max_abs_prediction_change_pct",
    ]
].round(3)
```

`export_rating_tables` writes the same payload as an Excel workbook: one sheet
of rating tables laid out side by side for a rater to key on, one for the
discretization impact above, and one for the model summary. The format comes
from the file suffix, so the path must name an `.xlsx` file rather than a
directory.

```{code-cell} ipython3
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as folder:
    workbook = Path(folder) / "rating_tables.xlsx"
    model.export_rating_tables(workbook, X, claims, offset=offset)
    with pd.ExcelFile(workbook) as book:
        print(workbook.name, "->", book.sheet_names)
```
