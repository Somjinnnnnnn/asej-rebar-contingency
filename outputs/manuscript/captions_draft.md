# Captions

Figure and table captions read by `build_manuscript_docx.py`, numbered as in the paper (tables in order of
first citation; Notes S1–S2, Tables S1–S23 and Figs. S1–S4 form the supplementary material). From rev8, rev7 Tables 1,
3, 6, 7, 11 and 15 are Supplementary Tables S15–S20, rev7 Tables 8 and 12 are covered by Tables S7 and S9, and rev7
Figs. 5, 7 and 8 are Supplementary Figs. S1–S3; from rev10, rev7 Table 10 (model risk) is Table S21.; from rev12, rev7 Table 1 (code junctions) is Table 1 again, the band table is covered by Table S8, and Table S15 holds the block-length sensitivity of the protection intervals.

## Figures

**Fig. 1.** Research framework of the study. The data prepared in Stage 1 feed the price-process models of Stage 2 for RQ1 and the quantile regressions of Stage 3 for RQ2, which are evaluated in the rolling-origin backtest of Stage 4; the long-horizon bands of Stage 5 for RQ3 are checked against the US steel price index, and Stage 6 for RQ4 converts the predictive distributions into allowances for the expenditure of a purchase schedule and tests their protection at past origins.

**Fig. 2.** TPSO monthly list price of SD40 12 mm deformed bar, January 2017 to September 2026, spliced across four item codes. (a) Nominal THB per ton and real price in THB of September 2026 (Bangkok CPI); vertical lines mark the months in which the item code changed. (b) Monthly change of the nominal price; the price moved in 90% of months.

**Fig. 3.** CRPS skill relative to the driftless random walk in the rolling-origin backtest (SD40 12 mm, monthly origins from June 2020 to August 2026), by forecast horizon. The numbers of scored forecasts are 75, 73, 70 and 64 at the 1-, 3-, 6- and 12-month horizons. Dashed line: the headline block bootstrap. Filled markers: Diebold-Mariano p < 0.05 before adjustment for multiple testing, none significant after the false discovery rate adjustment; hollow markers: p ≥ 0.05. Gray lines: the other models of Table 3, except the linear quantile regression with all drivers and a fixed penalty (−83% at 12 months).

**Fig. 4.** Empirical coverage in the backtest. (a) Share of outcomes at or below the P95 of the random walk and of the cross-validated XGBoost quantile regression, and below fixed allowances of +5% and +10%. (b) Coverage of the random-walk P5–P95 band for origins from June 2020 to December 2022 and from January 2023 on.

**Fig. 5.** One- to twenty-year price bands for SD40 12 mm from September 2026, a model-conditional scenario: headline band from a stationary block bootstrap (mean block 12 months) of demeaned real monthly log changes, with the P95 of the driftless real random walk as comparator. (a) Real THB of September 2026. (b) Nominal THB at 2% annual inflation. (c) USD at 2% inflation and an AR(1) path of THB/USD. Black: TPSO prices, 2017–2026.

**Fig. 6.** Worked procurement example. (a) Allowances for schedule A (600 t in eight quarterly lots over 24 months) from September 2026, nominal THB at 2% inflation: P80 and P95 of the expenditure ratio, quantity-weighted P95 of the purchase prices and P95 of the price at the last purchase; dashed lines mark fixed allowances of 5% and 10%. (b) Realized expenditure of the same schedule at past origins (nominal TPSO prices) against the P95 allowances of the random walk and the block bootstrap fitted at each origin, and a fixed +10% allowance.

## Tables

**Table 1.** Price change in the months in which TPSO changed the rebar item codes, compared with series whose codes did not change in that month. A junction is flagged when its change differs from the benchmark by more than two log points.

**Table 2.** Spliced TPSO rebar series, January 2017 to September 2026: coverage, share of months with a price change, nominal drift, volatility, and largest monthly rise and fall. Diameters with identical statistics are merged. The 180 ksc ready-mixed concrete series of the companion study is shown for reference.

**Table 3.** CRPS (% of the origin price) by model and horizon in the rolling-origin backtest, SD40 12 mm, including the headline block bootstrap of demeaned changes (mean block 12 months); in parentheses, skill relative to the random walk and the Diebold-Mariano p-value before adjustment for multiple testing. Unless marked "fixed", the quantile regressions are tuned by time-series cross-validation.

**Table 4.** Primary one-month comparisons of RQ2 (SD40 12 mm, 75 origins, quantile regressions tuned by time-series cross-validation): CRPS skill, Diebold-Mariano p-value and p-value adjusted by the Holm procedure over the four comparisons.

**Table 5.** Worked procurement example, model-implied allowances from September 2026, nominal THB at 2% inflation: percentiles of the expenditure ratio (% above the budget at the origin price), probability that fixed allowances of 5% and 10% suffice, and two price-based comparators, for schedule A (600 t in eight quarterly lots over 24 months) and schedule B (30 t at the end of each year for 20 years).

**Table 6.** Model-implied and demonstrated protection of schedule-A allowances at the 69 past origins from January 2019 to September 2024 (TPSO prices, models fitted at each origin to nominal changes or, in the rows marked "real fit", to real changes combined with 2% inflation): mean allowance and share of origins at which the realized expenditure stayed within it. The protection describes the allowance procedure refitted at each origin, not the allowances computed for September 2026. The scaled rows use the smallest factor k, computed from unrounded values, for which the scaled P95 allowance covers at least 95% of the 69 origins (66 of 69, or 0.96); they are a retrospective diagnostic derived from the same origins and have not been tested out of sample.

## Supplementary tables

**Table S1.** Item codes of the three 12 mm series: translated TPSO listing, unit, first and last month, and the check of each code change against the benchmark (SR24 median for 2018, median of the other category 04 steel items for April 2026; flagged when the difference exceeds two log points).

**Table S2.** Information available at forecast origin t: source, reference period, assumed release, lag with which each input enters, and data vintage.

**Table S3.** All exploratory Diebold-Mariano comparisons with the random walk (SD40 12 mm): CRPS skill, unadjusted p-value, and p-values adjusted by the Benjamini-Hochberg and Holm procedures over the 60 comparisons.

**Table S4.** Backtest on non-overlapping origins: every h-th origin for each of the h possible starting months; mean and range over the subsamples of the CRPS skill against the random walk and of the share of outcomes at or below P95.

**Table S5.** CRPS from the 19-quantile grid and from a 999-quantile grid, and the resulting skill against the random walk, for the models with closed-form or empirical quantiles.

**Table S6.** Empirical coverage of the P5–P95 band and of the P95 in the backtest for all models, including fixed allowances, and the random-walk coverage by period.

**Table S7.** Conformal recalibration for all models (per-quantile and symmetric forms, expanding and 24-month rolling calibration sets), evaluation origins May 2022 to August 2026.

**Table S8.** Price bands for SD40 12 mm from September 2026: P50, P80, P90 and P95 as % of the current price and as price levels, for the headline band and the comparator, in real THB, nominal THB under three inflation assumptions, and USD.

**Table S9.** Validation on the US PPI for steel mill products, all models, as fitted and with zero drift, in real and nominal terms: CRPS and coverage at 12, 36 and 60 months.

**Table S10.** Block-length sensitivity of the US PPI validation (real terms, demeaned block bootstrap): share of outcomes at or below P95, share in P5–P95 and CRPS at 12, 36 and 60 months.

**Table S11.** Dependence sensitivity: 1- to 20-year P95 (% above the current price) of the demeaned block bootstrap in real THB, nominal THB and USD when real rebar changes, Bangkok CPI inflation and THB/USD changes of 2017–2026 are resampled jointly in common blocks or independently (mean block 12 months, 10,000 paths).

**Table S12.** Basis of the junction flag: absolute monthly difference (log points) between each rebar series and the SR24 median in months without a code change.

**Table S13.** The demeaned block bootstrap in the 1–12-month backtest (SD40 12 mm), mean blocks of 12 and 1.4 months: CRPS, skill against the random walk with 90% stationary-bootstrap intervals, Diebold-Mariano p-value and coverage.

**Table S14.** Sensitivity of the schedule-A protection to the first evaluated origin. January 2019 is the first origin with 24 monthly changes for fitting; June 2020 is the first origin of the short-horizon backtest, with at least 41 monthly changes. The last origin is September 2024 in both cases. The scaling factor k is the smallest factor for which the scaled P95 allowance covers at least 95% of the origins in the set.

**Table S15.** Sensitivity of the 90% intervals for the protection of the schedule-A allowances at the 69 past origins to the mean block length of the stationary bootstrap of the origins; 24 months, the length of the overlapping outcome window, is the main choice.

**Table S16.** Diagnostics of the monthly log price changes (12 mm bars).

**Table S17.** Robustness of the CRPS skill relative to the random walk for the first-round models with fixed settings: one more month of publication lag for every driver, and the SD50 and SR24 12 mm series; in parentheses, the Diebold-Mariano p-value.

**Table S18.** CRPS skill relative to the random walk and share of outcomes at or below P95 in the backtest (SD40 12 mm), with 90% intervals from a stationary bootstrap of the forecast origins (2,000 resamples, mean block 2h months and at least 3). Quantile regressions tuned by time-series cross-validation.

**Table S19.** Sensitivity and uncertainty of the headline real P95 (% above the current price): block length, sample exclusions, comparator, Monte Carlo error (range over 20 seeds) and estimation uncertainty (median and 5th–95th percentile over 200 bootstrap histories). The last column gives the 60-month P95 coverage of the same specification in the US PPI validation, where available.

**Table S20.** Purchase profiles and an illustrative price-adjustment clause, 600 t over eight quarterly dates from September 2026, nominal THB at 2% inflation: expenditure P80 and P95, and, under a clause that passes each lot's price change beyond ±5% to the owner, the P95 retained by the contractor and the P95 passed on to the owner (% of the budget at the origin price); the two P95 values are not additive. Same simulated paths as Table 5 of the manuscript.

**Table S21.** Model risk: real price change after 5 and 20 years (P5 / P50 / P95, %), with drifts as fitted and with zero real drift.

**Table S22.** Characteristics of the forecasting and price-adjustment studies reviewed in Section 2 and of this study, as stated in the abstracts of the cited studies; "No" means that the characteristic is not addressed there.

**Table S23.** Protection of the schedule-A P95 allowance under alternative procedures, 69 past origins from January 2019 to September 2024: models fitted to nominal changes, as in Table 6 of the manuscript, with a 90% interval from a stationary bootstrap of the origins (mean block 24 months); models fitted to real changes with 2% inflation, as for the September 2026 allowances; non-overlapping origins 24 months apart; and a scaling factor estimated at each origin only from earlier origins with known outcomes (at least 12).

## Supplementary figures

**Fig. S1.** Conformal recalibration of the backtest bands. (a) Mean difference between the outcome and the random-walk median forecast, for the calibration origins (June 2020 to April 2022) and the evaluation origins (May 2022 on). (b) CRPS of four conformal variants relative to the raw random-walk forecast, evaluation origins May 2022 on; values above one are worse than no recalibration.

**Fig. S2.** Twenty-year change in the real price by model, with the drift of every model set to zero: P5 to P95 (line) and median (dot). Labels give the P95.

**Fig. S3.** Out-of-sample check of multi-year bands on the US producer price index for steel mill products (FRED WPU1017), deflated by the US CPI (CPIAUCSL): share of outcomes at or below the P95 of bands fitted to rolling 10-year windows, 141 monthly origins from January 2010 to September 2021, zero drift.

**Fig. S4.** Information timing of one forecast: reference month t of the TPSO list price, issuance of the forecast in month t + 1, target month t + h and observation of the outcome in month t + h + 1.
