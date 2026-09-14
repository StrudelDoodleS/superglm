# Named interaction candidates from primary sources

We now have eight concrete candidate entries across six datasets. Each has
raw column names and an explicit construction in the
[candidate manifest](../../benchmarks/known_interaction_candidates.json).
Three datasets have locally verified bytes. The competition candidates
remain references until their data are available through authorized access.
No fits, installations, or raw competition downloads were run for this review.

This review was made on 2026-09-14 in the adaptive-interactions worktree,
starting at `6340fdd8`. It supplements the
[earlier dataset evidence memo](2026-09-14-interaction-dataset-evidence.md).
The manifest is descriptive and does not extend a registry or launch jobs.

Evidence grades describe the source experiment, separately from the size
of its effect or whether it transfers to SuperGLM.

- A means an isolated held-out addition or removal of the named pair group.
  It does not assert statistical significance or universal validity.
- B means the named construction appears in a successful joint pipeline.
  Its individual contribution is unmeasured in the reviewed source.
- C means a fitted result or physical relationship motivates a hypothesis,
  including transfer to a different dataset.

| Candidate ID | Exact local or competition columns and construction | Grade | Data available locally |
| --- | --- | --- | --- |
| `bike_hour_workingday` | Cyclic spline basis of `hr`, multiplied by `workingday` | B | Yes |
| `ames_logarea_building_type` | `log10(Gr Liv Area)` multiplied by building-type indicators from `Bldg Type` | A | Yes |
| `porto_car13_ind03` | `ps_car_13 * ps_ind_03` and `ps_car_13 / ps_ind_03` | B | No |
| `porto_car13_reg03` | `ps_car_13 * ps_reg_03` and `ps_car_13 / ps_reg_03` | B | No |
| `home_credit_annuity` | `AMT_CREDIT / AMT_ANNUITY` | B | No |
| `home_annuity_income` | `AMT_ANNUITY / AMT_INCOME_TOTAL` | B | No |
| `ieee_amount_uid_mean` | Mean `TransactionAmt` within the joint key `card1`, `addr1`, `floor(TransactionDT / 86400 - D1)` | B | No |
| `concrete_water_cement` | `Water / Cement` | C | Yes |

The table includes several representation classes. A ratio is a function
of two raw columns, but a spline of that ratio and a two-margin tensor
express different functions. The IEEE construction uses several columns
and other rows. Its evaluation belongs in a separate aggregation experiment.

## What the sources establish

The scikit-learn Bike example reports MAE changing from 0.097 to 0.078
after adding its polynomial block to cyclic-spline ridge regression. It
models `count / 977` with squared error. Five time-series folds use a 48-row
gap, at most 10,000 training rows, and 1,000 test rows. The block includes
hour-spline products as well as hour-by-workday products, so grade B applies
to the named pair. Our proposed count-Poisson trial changes the link and
must establish the pair's value again.
[Original example](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html#modeling-pairwise-interactions-with-splines-and-polynomial-features)

Kuhn and Silge add `log10(Gr_Liv_Area) × Bldg_Type` to a linear Ames
model and report log10-price RMSE changing from 0.0803 to 0.0799. This is
an isolated mixed-pair group comparison, hence grade A, with a small mean
difference. It uses ten random folds within an 80% price-stratified random
training split. Our temporal/PID split and raw column names differ. The
manifest maps both predictors and target; coordinates in the book's
baseline are absent from our predictor contract.
[Comparison](https://www.tmwr.org/compare#creating-multiple-models-with-workflow-sets),
[split](https://www.tmwr.org/splitting),
[folds](https://www.tmwr.org/resampling)

Porto Seguro's second-place code enumerates all 15 pairs among six numeric
columns and creates a product and an oriented quotient for each. The two
listed pairs are the first two combinations. They enter a sigmoid neural
network trained with binary cross-entropy. The simplified NN/GBM ensemble
has a reported private normalized Gini near 0.2938; no pair ablation is
reported. The source uses five shuffled stratified folds, seed 218. These
products are in `nn_model290.py`; attributing a standalone LightGBM score
to them would be wrong. The code also uses combined train/test predictors
for some preprocessing, which needs separate handling in a frozen trial.
[Pair enumeration](https://github.com/xiaozhouwang/kaggle-porto-seguro/blob/master/code/nn_model290.py#L36-L38),
[exact arithmetic](https://github.com/xiaozhouwang/kaggle-porto-seguro/blob/master/code/util.py#L92-L98),
[reported ensemble](https://github.com/xiaozhouwang/kaggle-porto-seguro#simple-solution-recommended)

Home Credit competitor pklauke defines both named ratios in public code and
reports rank 248 of 7,198 for a blended pipeline with a binary-objective
LightGBM component. Its runner defaults to ten shuffled stratified folds
and repeated seeds. Neither ratio has an isolated AUC result. NoxMoon's
separate rank-17 writeup also names credit divided by annuity. The manifest
distinguishes the exact income denominator from the source's second
`1 + AMT_INCOME_TOTAL` variant.
[Feature code](https://github.com/pklauke/Kaggle-HomeCreditDefaultRisk/blob/master/feature_engineering.py#L221-L255),
[model code](https://github.com/pklauke/Kaggle-HomeCreditDefaultRisk/blob/master/models.py#L214-L283),
[competitor report](https://github.com/pklauke/Kaggle-HomeCreditDefaultRisk),
[independent competitor construction](https://github.com/NoxMoon/home-credit-default-risk#features-from-business-intuition)

IEEE winner Chris Deotte's coauthored account constructs the listed UID
and amount aggregates. It reports local AUC 0.9472 after adding a whole
UID feature bundle. The preceding evaluation uses the first 75% of rows
against the last 25%; the UID paragraph does not separately specify its
boundaries. This gives no isolated evidence for the amount mean. Before
testing it, specify which historical rows may contribute and when their
values are known. A pairwise spline cannot reproduce a cross-row lookup.
[Winner's account](https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution/)

Yeh's original augment-neuron study names water/cement ratio among seven
inputs. The accessible publisher abstract describes 100 random training
and 100 test examples, with no isolated ratio score or exact loss/link.
That 200-example study differs from our 1,030-row UCI concrete corpus.
The ratio is therefore a grade-C physical transfer hypothesis here. The
proposed local task uses raw compressive strength and Gaussian identity loss.
[Original modeling paper](https://ascelibrary.org/doi/10.1061/%28ASCE%290899-1561%281998%2910%3A4%28263%29),
[UCI schema](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)

GitHub citations above point to mutable branches observed on 2026-09-14.
The manifest records code locations and exact formulas for later checking.
Competition ranks and source metrics describe their original evaluation
protocols; they are not comparable with our temporal test scores.

## Which trials this permits

Bike, Ames, and Concrete source hashes match the current
[core registry](../../benchmarks/interaction_datasets.json). Home Credit and
IEEE are catalogued in the
[competition registry](../../benchmarks/interaction_kaggle_datasets.json)
with an existing HTTP 401 acquisition result from 2026-09-13. Porto Seguro
has no entry or acquired source in those registries. This review made no
new access probe. Availability means raw bytes exist, not that the new
recipes have an implemented runner.

Start with Bike and Ames using the precise constructions above. Concrete
offers a weaker external hypothesis on another locally available corpus.
For every trial, preserve the same target/link and adequate main effects
in both arms, then jointly refit after adding the candidate. Test a
source-exact engineered feature and a pair tensor as separate arms. Fix
zero-denominator and missing-value rules using training data. An additive
model given a ratio already has multivariable information in raw coordinates.

Use validation for representation choices and reserve an untouched test
for confirmation. The previously inspected Bike and Ames test results make
those tasks development evidence. Record all discovery and refit costs,
convergence failures, numerical outputs, and backend dispatch. A comparison
between constrained-additive, pairwise, and unrestricted GBMs under matched
loss and tuning helps distinguish interaction capacity from weak univariate
bases. These checks remain proposed work.

No verified Allstate pair emerged from this bounded primary-source review.
The earlier UCI credit and WDBC pairs retain their hypothesis status. The
bank does not relabel feature importance, a fitted plot, or competition
pipeline membership as established incremental pair utility.

## Locally measured follow-up

The subsequent [fixed broad trial](2026-09-14-broad-interaction-trials.md)
adds exact validation-selected pair groups on twelve new real sources and
one separate decoy control. Ten real sources select groups with lower test
loss; two retain additive models. Those local measurements complement this
published-reference bank and have their own frozen feature/split contract.

Concrete's training-only proposer admits `Cement × Water` fourth, alongside
three other pairs. The selected four-tensor model lowers test MSE by 16.68%
against its matching additive model. This recovers a physically motivated
variable pair, but does not isolate its contribution or test `Water / Cement`
as an engineered feature. The reference grades above remain unchanged.
