# Kaggle and public alternatives for interaction experiments

Date: 2026-09-13. Status: five real raw tables downloaded and validated;
modelling adapters and fit results are separate work. This extends the
[core corpus](2026-09-13-interaction-dataset-corpus.md) and the
[additive-normalized budget](2026-09-13-cheap-interaction-budget.md).
The machine-readable record is
[interaction_kaggle_datasets.json](../../benchmarks/interaction_kaggle_datasets.json).

The batch contains **11,954,804 raw rows in five tables**, delivered in
**210,657,215 bytes (200.90 MiB)**. Acquisition stayed below the authorized
1 GB compressed budget; the filesystem had about 1.8 TiB free beforehand.
Counts below describe complete raw tables. They are not eligible training
counts, independent observations, completed fits, or an accuracy result.

## Downloaded and verified

| Registry ID | Version and selected table | Rows × raw columns | Target and observed issues |
| --- | --- | ---: | --- |
| `kaggle_ulb_creditcard_fraud` | ULB v3, `creditcard.csv` | 284,807 × 31 | `Class`: 492 fraud, 284,315 other; no missing values |
| `kaggle_nyc_property_sales` | NYC v1, `nyc-rolling-sales.csv` | 84,548 × 22 | `SALE PRICE`: 14,561 missing, 10,228 zero |
| `kaggle_king_county_sales` | King County v1, `kc_house_data.csv` | 21,613 × 21 | `price`: positive, no missing values; 177 repeated-property rows beyond first occurrences |
| `kaggle_berkeley_city_temperature` | Berkeley Earth v2, `GlobalLandTemperaturesByCity.csv` | 8,599,212 × 7 | `AverageTemperature`: 364,130 missing |
| `kaggle_alternative_nyc_tlc_2024_01` | TLC January 2024 Parquet | 2,964,624 × 19 | `fare_amount`: 37,448 negative, 893 zero; off-month timestamps |

**ULB fraud** was downloaded from the Machine Learning Group–ULB publication,
which identifies real September 2013 card transactions from the Worldline
research collaboration. Its publisher-provided PCA coordinates are retained;
the original covariates and PCA fitting partition are unavailable. The manifest
selects V1–V28 and Amount, with Time reserved for chronological splitting.
Its published license is ODbL/DbCL. Use original prevalence in held-out data and
report average precision and log loss alongside ROC AUC and class counts.
[Publisher](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**NYC sales** comes from the City of New York publishing account, whose card
describes concatenated, lightly cleaned Department of Finance rolling-sales
files. The registry recognizes the exact blank and dash placeholders while
preserving the ZIP. Borough/block/lot identifies 67,239 property keys:
17,309 rows occur beyond the first observation of a key. Exclude row indexes,
addresses, unit identifiers, and fields explicitly labelled “AT PRESENT” from
the initial predictor set. Predeclare sale-date partitions, property-overlap
purging, and market-sale eligibility; zero or nominal transfers cannot silently
become market-price observations. The publisher declares CC0.
[Publisher](https://www.kaggle.com/datasets/new-york-city/nyc-property-sales),
[original municipal source](https://www.nyc.gov/site/finance/taxes/property-rolling-sales-data.page)

**King County** is a public community publication, described as real housing
sales during May 2014–May 2015. Its short publisher card does not document the
entire assessor-to-CSV chain, so retain that provenance qualification. The
publisher declares CC0. The manifest reserves property ID and sale date for
partitioning and selects the remaining property attributes. A row-random split
does not establish generalization to different properties.
[Publisher](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)

**Berkeley Earth** supplies a large temporal and geographic control: monthly
temperature estimates across 3,448 distinct city names and 159 countries.
Names do not uniquely identify locations; use country and coordinates too.
These are compiled estimates with temporal and geographic dependence.
The ZIP's five aggregation views count as **one source**, and only the city
member is registered for loading. Missing outcomes remain missing; date and
signed-coordinate parsing belong in a declared adapter. Temperature uncertainty
is excluded as a predictor. The publisher declares CC BY-NC-SA 4.0.
[Publisher](https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data),
[original data project](https://berkeleyearth.org/data/)

**TLC trips** were fetched from the official city portal's January 2024 yellow
taxi link. This is a public original-source alternative; its registry prefix
does not make it Kaggle-hosted. The file's bytes and hash pin this observation
because a monthly filename alone is not an immutable version. The prospective
benchmark described here is *retrospective fare prediction* from observed trip
distance, times, locations and rate fields. It does not describe advance booking
prices. Total payment and charge components are excluded. Pickup timestamps
span 2002-12-31 through 2024-02-01, so the adapter must record timestamp and fare
eligibility rules. The portal's terms and disclaimer are recorded without
inferring a separate public-domain license.
[Official TLC source](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

## Requested competitions without local access

Anonymous official file-list requests returned HTTP 401 for all five named
competitions. Kaggle credentials were absent from the checked environment
variables and default credential-file locations; no credential contents were
read. The installed client independently reported that Allstate file listing
requires authentication. No account login, new competition-rule acceptance,
or competition-file download occurred.

| Competition | Published modelling target or task | Current status |
| --- | --- | --- |
| [Allstate Claims Severity](https://www.kaggle.com/competitions/allstate-claims-severity/data) | Claim `loss` | Access blocked |
| [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) | `SalePrice` | Access blocked; original Ames is registered separately in the core corpus |
| [Home Credit](https://www.kaggle.com/competitions/home-credit-default-risk/data) | `TARGET`, with application and history tables | Access blocked |
| [Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit/data) | Original credit-risk competition | Access blocked; exact target column left unverified |
| [IEEE-CIS Fraud](https://www.kaggle.com/competitions/ieee-fraud-detection/data) | `isFraud`, transaction/identity tables | Access blocked |

The original CamelCase `GiveMeSomeCredit` slug must not be confused
with a later similarly named community competition. Catalogue entries leave
source-file hashes, versions, row counts, schemas and feature lists unset.
Authentication and any applicable rule acceptance remain distinct from installing
a client. Original Ames and its Kaggle subset must not be counted as independent
corpora.

[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) and
[BankSim](https://www.kaggle.com/datasets/ealaxi/banksim1) explicitly describe
simulated transactions. They are recorded as synthetic exclusions and do not
contribute to the real-source count. No unofficial competition mirrors were
used.

## Reproduction and tooling

The [shared corpus utility](../../benchmarks/interaction_datasets.py) accepts
this alternate manifest. A second Kaggle-specific downloader is unnecessary:

```bash
uv run python benchmarks/interaction_datasets.py list \
  --manifest benchmarks/interaction_kaggle_datasets.json
uv run python benchmarks/interaction_datasets.py fetch \
  --manifest benchmarks/interaction_kaggle_datasets.json --budget-mib 1024
```

It checks exact byte size and SHA256, reads only the declared ZIP member or
Parquet table, and validates ordered columns, row counts, missing counts and
target support. Available entries reserve their actual known byte length
against the download budget. Their expanded ZIP-member limits are also pinned.
Fetching the full manifest can allocate a large DataFrame for the city table;
the initial independent audit instead scanned 100,000 rows at a time.

The user subsequently authorized installation of the official
[Kaggle CLI](https://github.com/Kaggle/kaggle-cli).
[PyPI 2.2.4](https://pypi.org/project/kaggle/2.2.4/) was the current release
verified on this date. It was installed in the ignored research directory:

```bash
UV_TOOL_DIR="$PWD/.benchmark-artifacts/kaggle-cli/tools" \
UV_TOOL_BIN_DIR="$PWD/.benchmark-artifacts/kaggle-cli/bin" \
uv tool install --python 3.13 --no-config \
  --default-index https://pypi.org/simple kaggle==2.2.4
```

The executable is `.benchmark-artifacts/kaggle-cli/bin/kaggle`.
Smoke checks used an empty task-local
`KAGGLE_CONFIG_DIR=.benchmark-artifacts/kaggle-cli/config`:
version reporting succeeded; anonymous ULB file listing succeeded; Allstate
file listing returned authentication-required with exit status 1.
The installation receipt records all 32 resolved distributions, official
distribution hashes and exact commands. Global PATH, project dependency files
and production package versions were not changed. No skill pack was installed
or registered.

## Evidence and experiment boundary

Each downloaded folder under
`.benchmark-artifacts/interaction-datasets/<id>/` contains
`acquisition.json` and `validation.json`;
Kaggle sources also contain `publisher-metadata.json`.
The tracked registry records their hashes, raw-file identities, publisher
versions, exact schemas, selected features, exclusions and split specifications.
HTTP receipts retain content length, ETag and Last-Modified; signed redirect
query strings were not retained. Each request had a 30-second socket timeout,
an elapsed-time check against 180 seconds, and an exact expected byte cap.

All selected rows were scanned. Numeric columns were checked for infinities,
observed targets for numeric finiteness, and ZIP members were consumed through
their CRC check. Parsing used Python 3.13.14, pandas 3.0.5 and pyarrow 25.0.1.
The separate original access-probe receipt preserves every probe actually
made, including two additional housing-catalogue probes; only the five requested
competitions are registered here. Local raw readiness remains
`adapter_ready: false`.

The final integration check verified 21 artifact and receipt hashes, exact
manifest-to-scan schemas and target summaries, complete feature/exclusion
coverage, and all four local document links. The shared loader returned the
full King County table with 21,613 rows and 21 columns. Its source hash at this
check was `5755ebcb25ad99658c92b0d9c6d7de1bb679dbb0b66a365293ec6e548af2d56a`.

For interaction comparisons, lock each dataset's target definition, row
eligibility, split and feature-engineering recipe before viewing held-out
scores. Fit preprocessing and any target encoding only inside training.
Charge feature engineering and selection to the full pipeline, and compare
additive and interaction models on identical rows and parent features.
Report complete-fit time, peak RSS, convergence, actual dispatch and fresh
held-out metrics. Large table size alone does not establish breadth: use the
core corpus's separate housing, credit and cancer controls as well. This
acquisition task ran no model fits and makes no claim about automatic feature
engineering or interaction accuracy.

## Update 2026-09-19: eleven competition tables are now local

The five tables above came from public URLs. Eleven more come from Kaggle
competitions, which only hand out files to an account that has accepted each
competition's rules.

What happened, in order. The Kaggle CLI found a login token already sitting in
its default place; nothing here opened, copied or printed it, and the CLI reads
it by itself. Asking each of seventeen competitions for its *list* of files
worked every time. Asking for an actual *file* failed sixteen times out of
seventeen with "403 Forbidden" — the one that worked was the house-prices
tutorial competition, whose rules the account holder had accepted long ago.
The account holder then went to each competition page and accepted its rules,
after which the same download command worked. One competition,
`ClaimPredictionChallenge`, is closed to late joining, so its rules can never
be accepted and its files stay out of reach; that is why eleven competitions
are registered here and not twelve.

Each table below was downloaded once, checked byte-for-byte against a recorded
size and SHA-256, and then read through the shared loader, which also checks
the row count, the exact column list, the missing-value counts per column and
the target's support. All eleven passed. Together they add 2,799,567 rows in
842,256,652 bytes.

| Registry ID | Competition table | Rows | Target | Family | Bytes | SHA-256 prefix |
| --- | --- | ---: | --- | --- | ---: | --- |
| `kaggle_allstate_claims_severity` | Allstate Claims Severity | 188,318 | `loss` | gamma | 15,848,282 | `fe1dec6e5e9a` |
| `kaggle_bnp_paribas_cardif_claims` | BNP Paribas Cardif claims management | 114,321 | `target` | binomial | 51,820,265 | `cb7804341e1f` |
| `kaggle_give_me_some_credit` | Give Me Some Credit | 150,000 | `SeriousDlqin2yrs` | binomial | 2,791,537 | `af11b34ceace` |
| `kaggle_home_credit_default_risk` | Home Credit Default Risk, `application_train` | 307,511 | `TARGET` | binomial | 37,847,529 | `64289b17dd31` |
| `kaggle_homesite_quote_conversion` | Homesite Quote Conversion | 260,753 | `QuoteConversion_Flag` | binomial | 38,939,183 | `3d591f0330fe` |
| `kaggle_ieee_cis_fraud` | IEEE-CIS Fraud Detection, `train_transaction` | 590,540 | `isFraud` | binomial | 61,183,858 | `d426943b8100` |
| `kaggle_liberty_mutual_fire_peril` | Liberty Mutual Fire Peril Loss Cost | 452,061 | `target` | Tweedie p=1.5 | 580,956,011 | `d1944a670c51` |
| `kaggle_liberty_mutual_property_inspection` | Liberty Mutual Property Inspection | 50,999 | `Hazard` | poisson | 988,133 | `642cda7127b9` |
| `kaggle_porto_seguro_safe_driver` | Porto Seguro's Safe Driver Prediction | 595,212 | `target` | binomial | 31,424,352 | `e59be1048328` |
| `kaggle_prudential_life_assessment` | Prudential Life Insurance Assessment | 59,381 | `Response` | gaussian | 2,538,336 | `9db0613f05d3` |
| `kaggle_sberbank_housing` | Sberbank Russian Housing Market | 30,471 | `price_doc` | gaussian | 17,919,166 | `2221642984c0` |

Three of these tables need a note before anyone fits them. The Kaggle copy of
the Allstate file carries junk lines after row 188,318, so the registry stops
reading there. Fire Peril's target is zero in 99.7 per cent of rows and the
competition scored it with a per-row weight column, so both the smooth model
and the boosting control have to carry that weight. Sberbank's price is fitted
on the log scale with floor area as an offset, which is the price-per-square-
metre target its winners used.

Reading Fire Peril takes about a minute; the other ten are seconds.

The registry now records `availability: "kaggle_competition"` for these
eleven. The loader treats that like any other pinned artifact: if the file is
already there it is verified and read, and if it is missing it runs the pinned
Kaggle CLI to fetch exactly the one declared member, verifies it, and only then
publishes it.
