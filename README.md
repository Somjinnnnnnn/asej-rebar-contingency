# Probabilistic forecasting and contingency bands for Thai reinforcing steel prices

Data and code for the manuscript *"Probabilistic Forecasting of Reinforcing Steel Prices for Expenditure-Based Price-Escalation
Allowances: Evidence from Thailand"*
(submitted to Ain Shams Engineering Journal). `python run_all.py` reproduces every table and figure
and builds the submission Word file.

The companion study of ready-mixed concrete list prices (Ain Shams Engineering Journal, under
review, ASEJ-D-26-03178) has its own archive:
https://github.com/Somjinnnnnnn/asej-concrete-contingency (https://doi.org/10.5281/zenodo.23496701).
`data.py` and `models.py` are taken unchanged from that archive.

## What the code does

1. **Builds a continuous rebar price record.** TPSO (Trade Policy and Strategy Office, Ministry of
   Commerce) re-codes the same rebar product over time. `rebar_data.py` links the codes of each grade
   and diameter into monthly series for January 2017 to September 2026 and checks every code change
   against prices whose codes did not change (rule documented in the module docstring).
2. **Backtests 1- to 12-month probabilistic forecasts** of SD40 12 mm deformed bar at 75 monthly
   origins (June 2020 to August 2026): driftless random walk, GBM, empirical changes, AR(1), GARCH(1,1),
   AR(1)-GARCH(1,1), Merton jump-diffusion, Markov switching, and linear and XGBoost quantile
   regression with own-history features, scrap and iron ore, or all global drivers (iron ore, scrap,
   coal, Brent, Bangkok diesel, US steel prices, THB/USD), with fixed or time-series cross-validated
   regularisation. Scores: coverage, pinball loss, CRPS, Diebold–Mariano tests.
3. **Tests conformal recalibration** of the stored forecast quantiles.
4. **Projects 1- to 20-year contingency bands** from September 2026 in real THB, nominal THB (five
   inflation assumptions) and USD (fixed and stochastic exchange rate). Headline band: stationary block
   bootstrap of demeaned real changes; comparator: driftless real random walk.
5. **Validates multi-year bands** on rolling 10-year windows of the US producer price index for steel
   mill products, nominal and deflated by the US CPI.

## Quick start

Python 3.12 was used; 3.10 or later should work.

```
pip install -r requirements.txt
python run_all.py
```

The full run takes about 1.5–2 hours on a 16-thread laptop (Windows 11); `--workers` sets the number
of parallel processes and `--from N` restarts at step N. Every simulation uses a named, fixed random
seed, so a rerun gives the same numbers. The `outputs/` folder holds the results as reported in the
paper.

| Step | Command (run by `run_all.py`) | Output folder | Used for |
|---|---|---|---|
| 1 | `python audit_rebar.py` | `audit_2026-10` | Tables 1–3, Fig. 2 |
| 2 | `python check_drivers.py` | `audit_2026-10` | driver coverage, lead-lag correlations |
| 3 | `python backtest_rebar.py --out-dir backtest_main` | `backtest_main` | Table 5 |
| 4 | `python backtest_rebar.py --series "SD50 12mm" "SR24 12mm" --out-dir backtest_robust_grades` | `backtest_robust_grades` | Table 5 |
| 5 | `python backtest_rebar.py --extra-lag 1 --out-dir backtest_robust_lag1` | `backtest_robust_lag1` | Table 5 |
| 6 | `python backtest_rebar.py --out-dir backtest_v2 --driver-sets "all drivers" "scrap+ore" --tune both` (in shards) | `backtest_v2` | Tables 4, 6, Figs. 3, 4 |
| 7 | `python conformal_rebar.py --run backtest_v2` | `backtest_v2` | Tables 7, 7b, Fig. 5 |
| 8 | `python projection_rebar.py` | `projection_2026-09` | Tables 8, 9, Figs. 6, 7 |
| 9 | `python validate_long_horizon.py` and `--deflator CPIAUCSL --out-dir validation_wpu1017_real` | `validation_wpu1017`, `validation_wpu1017_real` | Table 10, Fig. 8 |
| 10 | `python revision_analyses.py`, `python revision2_analyses.py`, `python revision3_analyses.py` | `revision_2026-10` | bootstrap intervals, multiple testing (including the joint adjustment of 67 comparisons), procurement example, protection with dependence-aware intervals, real-fit procedure check, retrospective and earlier-origin scaling factors, first-origin sensitivity, sensitivity and dependence analyses (Tables 3–6, S3–S5, S10–S23, Fig. 6) |
| 11 | `python manuscript_tables.py`, `python revision_tables.py`, `python manuscript_figures.py` | `manuscript` | all manuscript and supplementary tables and figures (figure files fig_01–fig_09) |
| 12 | `python build_manuscript_docx.py` | `manuscript` | submission Word files: manuscript with Tables 1–6 and Figs. 1–6 embedded, supplementary Notes S1–S2, Tables S1–S23 and Figs. S1–S4, highlights |

Each backtest saves a checkpoint after every origin, so an interrupted run resumes where it stopped.

The table CSVs in `outputs/manuscript` are numbered in analysis order. In the paper the tables follow
the order of first citation: Table 1 is `table_02_code_junctions.csv`, Table 2 is
`table_01_data_audit.csv`, Table 5 is `table_06_robustness.csv` and Table 6 is
`table_05_backtest_coverage.csv`; the other numbers agree.

Step 12 takes the text from `outputs/manuscript/Manuscript_rebar_ASEJ_draft.docx` and the tables,
figures and captions from step 11, and writes `Manuscript_rebar_ASEJ.docx` and
`Highlights_rebar_ASEJ.docx` in the layout of the journal submission (A4, line numbers, single
spacing, native Word equations). It needs only the step 11 outputs, so `python run_all.py --from 12`
rebuilds the Word files in seconds. The manuscript text and the generated Word files are not archived
here, so `run_all.py` skips step 12 when the draft is absent; every table and figure comes from
steps 1–11.

## Repository layout

```
data.py, models.py        loaders and stochastic models from the companion archive (unchanged)
rebar_data.py             rebar item selection and code splicing
drivers.py                loaders for the global steel drivers in data_external/
rebar_models.py           random walk, block bootstrap, mean reversion, AR(1)(-GARCH), quantile regression, CV tuning
backtest.py               scoring helpers and process diagnostics
audit_rebar.py            step 1        check_drivers.py      step 2
backtest_rebar.py         steps 3-6     conformal_rebar.py    step 7
projection_rebar.py       step 8        validate_long_horizon.py  step 9
revision_analyses.py, revision2_analyses.py, revision3_analyses.py   step 10
manuscript_tables.py, revision_tables.py, manuscript_figures.py   step 11
build_manuscript_docx.py  step 12 (needs the manuscript text, not archived)
run_all.py                runs steps 1-12
data/                     raw TPSO, BOT and EPPO files, unmodified
data_external/            World Bank and FRED files, with SOURCES.md
outputs/                  results as reported in the paper
```

## Data

All raw files are included unmodified, as downloaded from their publishers.

| File(s) | Content | Source |
|---|---|---|
| `data/Price_Report m.yy.xlsx` (117 files) | Monthly construction-material price reports, central area (Bangkok), January 2017 – September 2026; rebar is category 04. File names use the Buddhist-era year (`1.60` = January 2017); the code reads the period printed in each report | Trade Policy and Strategy Office (TPSO), Ministry of Commerce, https://www.tpso.go.th/economic-data/price-struct |
| `data/CPIG_Bangkok.xlsx` | Consumer price index, Bangkok and vicinity (base 2023 = 100), deflator | TPSO, https://www.tpso.go.th/economic-data/consumer-price-index |
| `data/P09.xls` | Retail price of petroleum products in Bangkok; HSD B7 diesel | Energy Policy and Planning Office (EPPO), https://www.eppo.go.th |
| `data/Exchange_Rate 1–10.xlsx` | THB/USD monthly average buying (TT) and selling rates | Bank of Thailand, https://www.bot.or.th/en/statistics/exchange-rate.html |
| `data_external/CMO-Historical-Data-Monthly.xlsx` | Pink Sheet monthly prices (iron ore cfr spot, Australian coal, Brent), release of 2 October 2026 | World Bank, https://www.worldbank.org/en/research/commodity-markets |
| `data_external/FRED_WPU1017.csv`, `FRED_WPU1012.csv` | US PPI: steel mill products; iron and steel scrap | U.S. Bureau of Labor Statistics via FRED |
| `data_external/FRED_CPIAUCSL.csv` | US CPI, all urban consumers (validation deflator; October 2025 not published) | U.S. Bureau of Labor Statistics via FRED |
| `data_external/FRED_PIORECRUSDM.csv` | IMF global price of iron ore (loaded, not used in the models) | International Monetary Fund via FRED |

Retrieval dates, file checksums and row checks are in `data_external/SOURCES.md`. The rights to the
raw data remain with their publishers; they are redistributed here only so that the analysis can be
reproduced, and should be cited to the original source.

## How to cite

Please cite the paper and this archive (see `CITATION.cff`; the Zenodo DOI is given on the
repository page).

## Licence

Code: MIT licence (`LICENSE`). Results in `outputs/`: CC BY 4.0. Raw data in `data/` and
`data_external/`: terms of the original publishers (World Bank data: CC BY 4.0).
