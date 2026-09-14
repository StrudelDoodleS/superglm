# Fixed data protocol for thirteen additional interaction trials

All thirteen selected tables passed complete source loading and adapter preparation
before this batch's model trials on 2026-09-14 at 01:43:59 Europe/Berlin
(2026-09-13 23:43:59 UTC).
This batch contains twelve observational or measured-experiment sources and one
explicit CASP structural-decoy control. Preparation establishes usable inputs and
fixed partitions; it supplies no evidence about SuperGLM accuracy or convergence.
No response-model fits or final test evaluations were performed for this protocol.

The implementation is [broad_interaction_data.py](../../benchmarks/broad_interaction_data.py).
It uses the verified [core registry](../../benchmarks/interaction_datasets.json)
and [Kaggle registry](../../benchmarks/interaction_kaggle_datasets.json) through the
existing [corpus loader](../../benchmarks/interaction_datasets.py). Source provenance,
licenses and acquisition limits remain documented in the
[core corpus record](2026-09-13-interaction-dataset-corpus.md) and
[Kaggle corpus record](2026-09-13-interaction-kaggle-corpus.md).
King County is the pinned Kaggle publisher's version 1, with the provenance
qualification recorded there. CASP is excluded from counts of independent real
observational/experimental sources. Red and white wine are one combined table;
SGEMM's repeat responses are one experiment.

There are 435,555 source rows and 434,820 retained rows across the batch. Each
source is below the fixed 300,000-row admission limit. The adapter never samples
down to that limit, changes response scale, filters outliers or removes rows
because of a fit result. Every selected predictor survives the existing training
preprocessor in this batch. Here, predictor count means original input variables;
it is distinct from fitted coefficient count and smoothing-parameter count, as
described in the [scaling design](2026-09-13-real-data-scaling-design.md).

| Dataset ID | Raw target; family | Source rows | Predictors | Train | Validation | Test | Ineligible / purged |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `uci_abalone` | Rings; Poisson | 4,177 | 8 | 2,506 | 835 | 836 | 0 / 0 |
| `uci_concrete` | Concrete compressive strength; Gaussian | 1,030 | 8 | 643 | 173 | 214 | 0 / 0 |
| `uci_wine_quality` | quality; Gaussian | 6,497 | 12 | 3,924 | 1,276 | 1,297 | 0 / 0 |
| `uci_parkinsons` | total_UPDRS; Gaussian | 5,875 | 19 | 3,506 | 1,072 | 1,297 | 0 / 0 |
| `uci_protein_structure` | RMSD; Gaussian, exploratory decoy control | 45,730 | 9 | 27,438 | 9,146 | 9,146 | 0 / 0 |
| `uci_airfoil` | scaled-sound-pressure; Gaussian | 1,503 | 5 | 900 | 284 | 319 | 0 / 0 |
| `uci_power_plant` | PE; Gaussian | 9,568 | 4 | 5,740 | 1,914 | 1,914 | 0 / 0 |
| `uci_appliances` | Appliances; Gaussian | 19,735 | 24 | 11,562 | 3,888 | 3,997 | 0 / 288 |
| `uci_seoul_bike` | Rented Bike Count; Poisson | 8,760 | 11 | 5,208 | 1,728 | 1,529 | 295 / 0 |
| `uci_metro_traffic` | traffic_volume; Poisson | 48,204 | 7 | 26,675 | 10,728 | 10,801 | 0 / 0 |
| `uci_sgemm` | Run1 (ms); Gaussian | 241,600 | 14 | 83,168 | 87,264 | 71,168 | 0 / 0 |
| `kaggle_king_county_sales` | price; Gaussian | 21,613 | 18 | 13,738 | 3,083 | 4,640 | 0 / 152 |
| `uci_superconductivity` | critical_temp; Gaussian | 21,263 | 81 | 12,854 | 4,244 | 4,165 | 0 / 0 |

Random group partitions use the existing `partition_rows` implementation with
seed 20260913 for the 60/40 group split and seed 20260914 for the remaining 20/20.
The fractions refer to groups, so unequal group sizes can give very different row
fractions. SGEMM, for example, has only sixteen MWG/NWG groups. No alternative seed
or regrouping is chosen to improve its row balance or observed performance.

| Dataset | Partition unit and interpretation |
| --- | --- |
| Abalone | All eight predictors define an identical-profile group: 4,177 groups. Cross-sectional exploratory control; collection-location IDs are absent. |
| Concrete | Seven mixture ingredients define 427 groups across curing ages. Age remains a predictor. Laboratory batch IDs are absent. |
| Wine quality | All predictors, including color, define 5,320 identical-profile groups. Winery/batch IDs are absent. |
| Parkinsons | All recordings from each of 42 subjects stay together. The related motor_UPDRS response and subject ID are excluded. Labels are supplied interpolated longitudinal scores. |
| CASP | Explicit zero-based source-row positions define the random units. Protein IDs are absent, so protein-independent transfer cannot be claimed. |
| Airfoil | Attack angle, chord length and free-stream velocity define 106 tunnel-configuration groups across frequencies. The response already uses dB; no further logarithm is taken. |
| Power plant | Four predictors define 9,527 identical-profile groups. The shuffled source has no observation timestamps and does not support a prospective forecast claim. |
| SGEMM | Sixteen MWG/NWG tiling groups test configuration transfer. Run2, Run3 and Run4 are excluded responses; Run1 stays on its original millisecond scale. |
| Superconductivity | 15,542 exact material-formula groups stay intact. The verified loader aligns the companion material column using the supplied critical_temp column. This is formula transfer, not a stronger chemical-family holdout. |

For all four chronological tables, the adapter parses the known format before
sorting, normalizes timestamps to calendar days, and takes the first
`floor(0.6 * observed_days)` days for training, the next days through
`floor(0.8 * observed_days)` for validation, and the remaining days for test. Days
absent from the source are not invented. Every row of one observed day receives
the same initial partition. Cutoffs are fixed using all source rows, before
operating eligibility, missing-target exclusion or boundary purges.

| Dataset | Exact source spelling | Raw training days | Raw validation days | Raw test days |
| --- | --- | --- | --- | --- |
| Appliances | `%Y-%m-%d%H:%M:%S`, e.g. `2016-01-1117:00:00`; the verified UCI CSV has no date/time separator | 2016-01-11–2016-04-01 | 2016-04-02–2016-04-29 | 2016-04-30–2016-05-27 |
| Seoul bike | Day/month/year, e.g. `1/12/2017`; optional leading zeros are normalized explicitly, then parsed as `%d/%m/%Y` | 2017-12-01–2018-07-07 | 2018-07-08–2018-09-18 | 2018-09-19–2018-11-30 |
| Metro traffic | `%Y-%m-%d %H:%M:%S`, e.g. `2012-10-02 09:00:00` | 2012-10-02–2016-09-16 | 2016-09-17–2017-09-23 | 2017-09-24–2018-09-30 |
| King County | `%Y%m%dT%H%M%S`, e.g. `20141013T000000` | 2014-05-02–2014-12-11 | 2014-12-12–2015-03-03 | 2015-03-04–2015-05-27 |

The Appliances manifest calls for a one-day gap. The adapter removes the one
calendar day immediately before each later partition: 144 rows on April 1 and
144 rows on April 29. Retained training therefore ends March 31 at 23:50, and
retained validation ends April 28 at 23:50. This is prediction at later times in
one building. Date, the concurrent lights response and random rv1/rv2 distractors
remain excluded predictors.

The Seoul manifest declares an operating-demand estimand. `Functioning Day=Yes`
eligibility is applied after the whole-day split. It excludes 48 closed training
hours, 24 closed validation hours and 223 closed test hours, for 295 total.
Neither the date nor the operating flag enters the features. Unexpected or
missing operating labels fail preparation. Closed hours are not reallocated and
their removal does not move the original September 19 test boundary; the first
retained test day happens to be September 20.

Metro has 48,204 weather annotation rows for 40,575 distinct timestamps. All
annotations at one timestamp were checked to have exactly the same traffic
count. Each receives weight `1 / number_of_annotations_at_that_timestamp`, so
each observed hour contributes total weight one. The train/validation/test
effective-hour counts are 22,829 / 8,846 / 8,900. All annotations stay in their
whole-day partition. Both fitted systems and any validation loss must receive
these weights. This estimates loss averaged across annotations within each hour;
the adapter does not create an aggregated weather representation. Literal
holiday `None` remains a category, as specified by the source parser. Date/time
is excluded from predictors.

King County contains 21,436 property IDs. A property is assigned to its latest
original chronological partition, and its earlier-partition observations are
purged. This removes 152 rows belonging to 151 crossing properties: 135 training
rows and 17 validation rows. It preserves chronology, keeps retained property
IDs disjoint and moves no future sale into an earlier partition. Repeated sales
within the assigned partition remain. The target never determines which
occurrence is kept. ID and sale date remain excluded predictors. This fixed
purging rule conditions the sample on the observed sale record; it is not a
claim of an untouched prospective population.

Every source row is accounted for exactly once as retained, ineligible or
purged. All thirteen pinned tables have observed finite targets. Unexpected
missing targets are errors under their existing manifests. For fixture or future
entries that explicitly set `target_rule.allow_missing=true`, missing targets
are removed after initial partitioning and never imputed. Infinite targets,
values outside declared support and fractional/negative Poisson counts fail
preparation instead of triggering row deletion. Missing group/time keys also
fail preparation. Invalid feature values are not converted into row exclusions.

The existing `fit_preprocessor` is reused without changes. It sees only retained
training rows and precisely the manifest's feature list and categorical hints.
Numeric imputation medians, centering and scaling, constant-column removal,
numeric versus spline classification, and category pooling all depend on that
training frame. Its category cap remains 32. Validation and test rows are later
transformed with this frozen state. Preprocessing statistics use the original
training-row convention; Metro fitting and losses use the supplied hour weights.
All target, related response, identifier and other excluded columns are checked
against the predictor list. The raw frame retains its source columns and row
order; no helper split column enters the model.

The parent-facing API is:

```python
prepared = load_prepared(dataset, data_root=DEFAULT_ROOT)
# keys: entry, frame, state, rows, sample_weight, metadata
entry = prepared["entry"]
train_rows = prepared["rows"]["train"]  # original source positions
train = prepared["frame"].iloc[train_rows]
X = transform_features(train.loc[:, entry["features"]], prepared["state"])
y = response_values(train, entry)  # original response scale
w = prepared["sample_weight"][train_rows]
```

`prepare_frame(frame, entry)` exposes the same preparation boundary for fixtures
and does not claim source-byte verification. `load_prepared` first uses the
existing loader's complete hash/schema validation and then marks source bytes
verified. Metadata records the source registry/path, exact entry and source
identities, actual and initial partitions, original positions excluded by each
rule, weights, training state, and source-code hashes. The adapter sets no basis
resolution, pair proposal budget or fitting policy. Test positions are reserved;
this protocol does not consume their model losses.

The full local receipt is
`.benchmark-artifacts/broad-interactions/data-protocol/20260913T234359-preparation.json`,
SHA256 `e1e2bfb5365f64b8ee9a281e5f40e66eb0045aad77f43b460b88da96e686abe0`.
It contains all training states and exact exclusion positions. It was written
atomically without replacing an existing receipt. The durable snapshot below
retains the identities and compressed exclusion positions in this tracked file.
All position ranges there are zero-based and inclusive. Repeating preparation
from the pinned bytes and registry reproduces the retained row and state hashes.

These are the measured source and registry hashes from preparation, preserved
in source checkpoint `639f499e`. The frozen adapter source SHA256 is
`6bde2ec96fb97ecc888aa3dd4e7c1d9a22eafbe694864a17fa83984abe1eacba`.
The reused source-loader SHA256 is
`c28aa62c9a154405fd901c3c1898dd0c899742f17a202da6b2c46aa82cabf3d0`;
the reused real-trial/preprocessor module SHA256 is
`d1fd02b97005d0a71c094dec05488343f5a712c821ae7866d5227d4ec1a16b13`.
The core registry SHA256 is
`8ad0bbac4dbdf0580d15a6bafa50d7147c48ddeb654e75642f60ab944c8bcbb3`;
the Kaggle registry SHA256 is
`59ecbd64ac86813866e027b1ec7769a0775232d6b2ddc28cd1dacb33e3eab2ac`.
Later PR-review corrections changed receipt handling in
`benchmarks/benchmark_real_interactions.py`, so its current module hash differs
from the measured value above. The adapter and source-loader files are unchanged.
Replay uses the [frozen source and environment](2026-09-14-interaction-review-validation.md#replaying-a-historical-source-tree),
not the corrected launcher module in place of its measured version.
Preparation used Python 3.13.14, NumPy 2.5.2, pandas 3.0.5 and scikit-learn 1.9.0.
No environment or dependency files were changed.

Focused validation is
`uv run pytest benchmarks/test_broad_interaction_data.py benchmarks/test_real_interactions.py -q`
(43 passed), plus Ruff check and format check for the two new Python files.
Tests include a demonstration that the older raw-string chronological split
orders Seoul dates incorrectly, failing regressions for the actual UCI date
spellings before the parser fixes, complete row accounting, boundary purges,
Metro duplicate-hour weighting and target consistency, material/subject group
separation, target exclusions and held-out changes that cannot alter training
state. Full preparation succeeded on every selected table; no timed response
fit was used as a data-admission check.

<!-- The following snapshot is generated from the complete preparation receipt. -->

```json
{
  "uci_abalone": {
    "source_url": "https://archive.ics.uci.edu/static/public/1/data.csv",
    "data_sha256": "25f3dc964447fcb3d68cf1048c04ed69d1474771c4fb34318e10778800a4342f",
    "manifest_entry_sha256": "751b3ab706a02e48ad2d03a1ec52a3ec462e8fa513dfb6339773c2678ca826c0",
    "split_sha256": "1bb2a4613d858bd16173d141b13e3c5667e7ecd3923291ab7888f16abbd9004a",
    "preprocessing_sha256": "583c67f6cb4b5919e3ff32e50a74809f85de6d80c967a2e0bde11ef9048239fa",
    "sample_weight_sha256": "e3320341a948e13582f896ef0a191e1b97fcd35c6ccf422395b20d797fa36cb6"
  },
  "uci_concrete": {
    "source_url": "https://archive.ics.uci.edu/static/public/165/data.csv",
    "data_sha256": "8d4b15b6fc68cd932d745cbd663d5ceae66dd54422e99c1e4865f2936ab7e2af",
    "manifest_entry_sha256": "ab335d2d71bddeacb26c5a18ea95991a0be8d025db97629a0e3182316c8c16aa",
    "split_sha256": "a1b0ffb1e775d47fdecb2198363c20be4363b0c93d27e11d1fa12ea03d8bdcb1",
    "preprocessing_sha256": "5693ec3701237df4e1c6f34a143f2297d53243edcf69c63904b209b684cc9bdb",
    "sample_weight_sha256": "ab566fea45f4ef624782a4218fc4f110989b8e8e9d25bb1e8c94d7dcf3109a92"
  },
  "uci_wine_quality": {
    "source_url": "https://archive.ics.uci.edu/static/public/186/data.csv",
    "data_sha256": "428bd8df7313b159988405ae5e3be6d363993d73f7ffc2adedfdc1d496508191",
    "manifest_entry_sha256": "1889bb605bbb0860dd7b24a83d4c820953e434f2b1d5aeda2d0193133c5c0b2a",
    "split_sha256": "cba30339ec6b522d3a409de73a5c023438a9699352870c2cad10684f4a811e2b",
    "preprocessing_sha256": "8820613b4259c7f8efcba9cc6ca6ff133fc2b4b3ef1f2100436a3bce17287990",
    "sample_weight_sha256": "92516eca59c02d0b42ea782ba8b09f8c0d3c8616409fc95b66457efa286f82e5"
  },
  "uci_parkinsons": {
    "source_url": "https://archive.ics.uci.edu/static/public/189/data.csv",
    "data_sha256": "689fd221159cd78b07a50087992c9d9472a4eb26527cdc8e896a708a2013ebcf",
    "manifest_entry_sha256": "5c8ba268deec629019aadb43bf7ad52eee5a22b9e6e564b1220ff600b9edf119",
    "split_sha256": "682b9dbb46b203ae6442429f2bcc87c52e8f54e6d87fc67b2357d87a3c5e64c2",
    "preprocessing_sha256": "bc7a01124d983f4459c1629e6452aefb293fe391b1f6aec82405f26ad4e9f3f7",
    "sample_weight_sha256": "03575678d22e5f349c4ab30b057ff4e864e0d60dff9d02a58287559afd714a34"
  },
  "uci_protein_structure": {
    "source_url": "https://archive.ics.uci.edu/static/public/265/physicochemical+properties+of+protein+tertiary+structure.zip",
    "data_sha256": "ee6536c8cc415dc50d66bd247af7c9bc95c187ab8a4ea0f1f546b127206c1100",
    "manifest_entry_sha256": "9dc2bbf6b9317b034c95c47915d1accf7b6fa244dc577e4fad975179b239ef44",
    "split_sha256": "1010a4295e0115110e01b770a1359ceff020b611cbef7a13b313dccfbae29df0",
    "preprocessing_sha256": "1247f32dbc8c977220f74bbfb0c005767303501c2b86df16bccde681948c1d7e",
    "sample_weight_sha256": "6586eb5234bd8d354e6f16227d6a679d4f7c7fe861a93d96e8900eb64932699c"
  },
  "uci_airfoil": {
    "source_url": "https://archive.ics.uci.edu/static/public/291/data.csv",
    "data_sha256": "0beffa51c5b1493960d046c7d512a67d06a7884df6db599408087ee2fc2fc18a",
    "manifest_entry_sha256": "e6dc9a2e596d7700d4e19269a1141e7e2bdfa544f46cfc4e847f807910b66be8",
    "split_sha256": "00065ecd25696ced1c713acfe964e8a9fbe13bbf421afc100c70e844eed04081",
    "preprocessing_sha256": "0abdfa22de18ed4229be1644366bf7a59badac7447e84b4f905b2b7ddd04a4e6",
    "sample_weight_sha256": "2d48e5dfa9de9601df9327789f82995d0a6f954b913450ca81428585798677e5"
  },
  "uci_power_plant": {
    "source_url": "https://archive.ics.uci.edu/static/public/294/data.csv",
    "data_sha256": "3c1fc11025f8424f8d95802d8b7086dffd3f73a552c6dcab3d973620986194b2",
    "manifest_entry_sha256": "bbb9d0b5feb40abed4178ad24bd7b62cd93926ba75b38abe0c2dca2cdcab8fbb",
    "split_sha256": "0dc82b7db4c8d1d60d94adf024f969824cecd1af8d75100188180a93dea99b97",
    "preprocessing_sha256": "428646995cff92f43a5942cc6e3ceb7bd4da9a57704f4576a9a7f43a0c472a59",
    "sample_weight_sha256": "adb1f97f8b2bd703853a9ba5e16b6f649f9e2d7414e92a341e7d52870cfe44c6"
  },
  "uci_appliances": {
    "source_url": "https://archive.ics.uci.edu/static/public/374/data.csv",
    "data_sha256": "df49e914acd72504f1109bd9fdb4a6970cb0effc8c7e977c92f95295af350e24",
    "manifest_entry_sha256": "f275c6c1773b8b95293d11903502fe394bbc00dcb11f33f1077d6d3ea274c0f3",
    "split_sha256": "f025cea8f9c8ebf496f9ae393c4a7cb75c28afb8bd140d83ccc77f732bfed4af",
    "preprocessing_sha256": "9f391019a001c44882ac8eff55855ef2ea9b19b48f1cc07e2f4844348c9ceb81",
    "sample_weight_sha256": "34c942efc13c5823ebd0e3d62f8a6724e148e915e023f59f4813c483659cf7f3"
  },
  "uci_seoul_bike": {
    "source_url": "https://archive.ics.uci.edu/static/public/560/data.csv",
    "data_sha256": "e392e3be042e7c0e58b48329a777206acb7f3400da95ee7cf405da5eb99d283e",
    "manifest_entry_sha256": "07ad0dab245a64ab4bdd72dffc6e921107ec122cee9f0fc9f7583732997d8be9",
    "split_sha256": "ea2056a53e7f22be6bd7deef15ba4de69fc294bd20377114d6586d965ed41dd8",
    "preprocessing_sha256": "bfc7d15cf2c73ea39cd8c7eee213361046c6404e3f4499b0366e86ce4e35e63e",
    "sample_weight_sha256": "c5c2c006f733e34ed0748a363bc049e58a4e79c35ce592f6f70788c266a89a66"
  },
  "uci_metro_traffic": {
    "source_url": "https://archive.ics.uci.edu/static/public/492/data.csv",
    "data_sha256": "749c90d720360a4215bb15345526073c079ba4cc95e3fa558796d083f85fce9e",
    "manifest_entry_sha256": "d2b611941ac622c4989223015dd851a5a14930c4ecf9cd44a1848f6a4e0e4b25",
    "split_sha256": "4092e564511346fc12d6fbd9fd1f9799e9579d9d52d82817b78e9acf3d300b74",
    "preprocessing_sha256": "db56d8bb4a4d575adc15e23e73622cf05b29c288ecc7f7975078e1434a7c6912",
    "sample_weight_sha256": "a1270ccd765a7d157f8b0ca94bab8cec703d2d3c37e0ddab290d6a6801a9fb84"
  },
  "uci_sgemm": {
    "source_url": "https://archive.ics.uci.edu/static/public/440/sgemm+gpu+kernel+performance.zip",
    "data_sha256": "8d3ee0d82708df11b54debc6a4fbade77c604a047f019fa7940413032eb7b096",
    "manifest_entry_sha256": "7c8e7391187b7d484176d897352407a5ec517d25439a9d34795c312479dc38ee",
    "split_sha256": "c1c1077b6865135c5dda90eb58c77098b196ce2f580d32c93a2721d648d645a8",
    "preprocessing_sha256": "7a67ac11638338bb0496163f3040775ae803d1b6009e2c9c313098bd3dc2cff6",
    "sample_weight_sha256": "c8d106e25a66d6cb4fdf553ae5ffc0bd65c6a5a870a5118da06633c64059bf0a"
  },
  "kaggle_king_county_sales": {
    "source_url": "https://www.kaggle.com/api/v1/datasets/download/harlfoxem/housesalesprediction?datasetVersionNumber=1",
    "data_sha256": "1fdffaeec5e7656f6c09e29d9daa7e2157479c5f8e3a761758b4bfa43c773202",
    "manifest_entry_sha256": "2d3cfed694b3a70a12f5afdc2d79c3c8d0e1a8485bbd504afcc1363f92ea07bc",
    "split_sha256": "6b362c547031918276fa06179d39d063217650568b4339d8538114d9f77bf569",
    "preprocessing_sha256": "dded9323b5aaa596461a70ef82b0ef6a2fe4687b022ff20128d0195edb685938",
    "sample_weight_sha256": "bb8ba14e95a02174bb0e1d54a485740803b2f52d2251ba584956bc4a0c4399e9"
  },
  "uci_superconductivity": {
    "source_url": "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip",
    "data_sha256": "87f4490d73390ff94ee01dbf0d7d32abc80b22f2c803d471765cfc46a9f6371e",
    "manifest_entry_sha256": "77ab6a7c71e1dca09a7d5651e6e8a802758d7b5c7deb5e3bda43b7dcd9306539",
    "split_sha256": "ed21805eab993b2ed91619a2932b2cbb8ff9a978124fde73facce7f6cfdce336",
    "preprocessing_sha256": "45859f55f6627f2de40bda4b93e1f4745ae10af92f30c60cda414873d8bbdeb4",
    "sample_weight_sha256": "65489f25c26f1fb40ddc1b5967e6dd67e8e905db25b15d0ec7e3683c27c55072"
  }
}
```

Exact excluded positions, compressed into inclusive ranges:

```json
{
  "uci_appliances": {
    "one_day_before_valid": [[11562, 11705]],
    "one_day_before_test": [[15594, 15737]]
  },
  "uci_seoul_bike": {
    "closed_hours": [[3144, 3167], [3840, 3863], [6984, 7031], [7224, 7247], [7272, 7295], [7320, 7343], [7368, 7391], [7416, 7422], [7488, 7511], [8088, 8111], [8160, 8183], [8232, 8255]]
  },
  "kaggle_king_county_sales": {
    "property_seen_in_later_partition": [[93, 93], [324, 324], [345, 345], [371, 371], [717, 717], [823, 823], [836, 836], [1085, 1085], [1128, 1128], [1202, 1202], [1234, 1234], [1450, 1450], [1464, 1464], [1576, 1576], [1864, 1864], [2038, 2038], [2126, 2126], [2493, 2493], [2496, 2496], [2502, 2502], [2531, 2531], [2564, 2564], [2631, 2631], [2976, 2976], [3033, 3033], [3298, 3298], [3540, 3540], [3623, 3623], [3756, 3756], [3793, 3793], [3878, 3878], [4077, 4077], [4342, 4342], [4872, 4872], [4922, 4922], [5244, 5244], [5272, 5272], [5340, 5340], [5592, 5592], [5694, 5694], [5723, 5723], [5756, 5756], [5972, 5972], [6345, 6345], [6371, 6371], [6434, 6434], [6533, 6533], [6630, 6630], [6719, 6719], [6789, 6789], [6902, 6902], [7178, 7178], [7245, 7245], [7792, 7792], [7847, 7847], [8011, 8011], [8125, 8125], [8274, 8274], [8340, 8340], [8433, 8433], [8505, 8505], [8532, 8532], [8630, 8630], [8904, 8904], [8915, 8915], [9016, 9016], [9114, 9114], [9120, 9120], [9234, 9234], [9275, 9275], [9277, 9277], [9279, 9279], [9392, 9392], [9439, 9439], [9489, 9489], [9506, 9506], [9720, 9720], [9820, 9820], [9876, 9876], [10253, 10253], [10272, 10272], [10319, 10319], [10610, 10610], [10969, 10969], [11061, 11061], [11194, 11194], [11202, 11202], [11287, 11287], [11775, 11775], [12030, 12030], [12065, 12065], [12121, 12121], [12332, 12332], [12338, 12338], [12388, 12388], [12434, 12434], [12832, 12832], [12920, 12920], [12955, 12955], [13023, 13023], [13183, 13183], [13298, 13298], [13617, 13617], [13628, 13628], [13692, 13692], [13734, 13734], [13756, 13756], [14220, 14220], [14307, 14307], [14366, 14366], [14484, 14484], [14574, 14574], [14855, 14855], [14982, 14982], [14993, 14993], [15146, 15146], [15156, 15156], [15199, 15199], [15277, 15277], [15299, 15299], [15468, 15468], [15607, 15607], [15656, 15656], [15801, 15801], [15810, 15810], [16000, 16000], [16012, 16012], [16671, 16671], [16814, 16814], [17066, 17066], [17133, 17133], [17280, 17280], [17290, 17290], [17381, 17381], [17482, 17482], [17580, 17580], [17602, 17603], [17763, 17763], [17851, 17851], [18564, 18564], [18703, 18703], [18705, 18705], [18762, 18762], [18991, 18991], [19209, 19209], [19335, 19335], [19552, 19552], [20180, 20180], [20669, 20669], [20779, 20779], [21580, 21580]]
  }
}
```
