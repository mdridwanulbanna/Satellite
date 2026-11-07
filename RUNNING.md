# NBoB satellite SST reliability and forecast skill: analysis code

Station coordinates and the LSTM set-up are taken unchanged from (Rifat et al. 2026) so that
all results are comparable with the published forecasting work.

## Install

```
pip install -r requirements.txt
copernicusmarine login          # once, for Step 1b
```

## Data you need to download yourself

| Data | Where | What to get |
|---|---|---|
| iQuam v2.10 in situ SST | NOAA STAR iQuam "Data" page (FTP link) | Monthly netCDF files. Start with 2023-01 to 2024-12 (24 files), extend to 2019 if counts are low. Files are global, so the download is large. |
| GEBCO_2023 bathymetry | GEBCO grid download app | netCDF subset for 18-23 N, 86-93 E |
| C3S / ESA CCI L4 SST | Copernicus Marine (downloaded by the script) | dataset `C3S-GLO-SST-L4-REP-OBS-SST`, variables `analysed_sst`, `analysed_sst_uncertainty`, `mask` |

## Run order

```
# Step 1a: in situ data density (Part A feasibility)
python step1a_iquam_scoping.py --inspect data/iquam/<one_file>.nc      # check variable names first
python step1a_iquam_scoping.py --iquam-dir data/iquam --gebco data/gebco_nbob.nc --out out/step1a

# Step 1b: station L4 series and L4 uncertainty (also produces the input for Step 2)
python step1b_l4_station_series.py --out out/step1b
#   or, with a file you already have:  --local-file my_subset.nc

# Step 1c: L4 analysis uncertainty (analysis_error_sst from the C3S dataset)
python step1c_l4_uncertainty.py --out out/step1c

# Step 4: L3 subsets and inventory
python step4_l3_download.py --product aqua --inspect
python step4_l3_download.py --product metopb --out OUT --workers 4   # also aqua, terra
python step4_l3_download.py --inventory --out OUT

# Step 5: RQ1 reliability metrics, screening and scoring (after Step 4; Colab: Step5_reliability_Colab.ipynb)
python step5_reliability.py --l3 L3DIR --iquam IQDIR/box_monthly --out OUT5
python step5_reliability.py --l3 L3DIR --iquam IQDIR/box_monthly --l4 OUT5/l4_daily_stationarea.nc --bathy OUT5/bathymetry_nbob.nc --ql-min 5 --out OUT5/check_ql5
python step5_reliability.py --l3 L3DIR --iquam IQDIR/box_monthly --l4 OUT5/l4_daily_stationarea.nc --bathy OUT5/bathymetry_nbob.nc --radius-km 10 --out OUT5/check_r10

# Step 5b: accuracy fallback (plan v1.0, Section 3): offshore units with < 50 matchups in 2018-2020 get the
# 2016-2017 matchups of the same phase (Colab: Step5b_accuracy_fallback_Colab.ipynb downloads them)
python iquam_scope_auto.py --start 2016-06 --end 2016-09 --out IQEXT; python iquam_scope_auto.py --start 2017-06 --end 2017-09 --out IQEXT
python step4_l3_download.py --product aqua --start 2016-06-01 --end 2017-09-30 --months 6 7 8 9 --out L3EXT   # also metopb, terra
python step5_reliability.py --l3 L3EXT --iquam IQEXT/box_monthly --bathy OUT5/bathymetry_nbob.nc --start 2016-06-01 --end 2017-09-30 --out OUT5EXT
python step5_reliability.py --l3 L3EXT --iquam IQEXT/box_monthly --bathy OUT5/bathymetry_nbob.nc --l4 OUT5EXT/l4_daily_stationarea.nc --start 2016-06-01 --end 2017-09-30 --ql-min 5 --out OUT5EXT/check_ql5
python step5b_accuracy_fallback.py --main OUT5 --ext OUT5EXT --out OUT5FINAL      # Step 6 then reads --step5 OUT5FINAL

# Step 6: link analysis (analysis plan v1.0 approved 11 Oct 2026)
python step6_link_analysis.py --synthetic --out /tmp/test6
# stage 1 (done): test 1 and family A, no Level-3 data needed
python step6_link_analysis.py --signed-off-on 2026-10-11 --hindcast results/step2/hindcast_station_month_lstm5seeds.csv \
    --uncertainty results/step1c/station_monthly_l4_uncertainty.csv --out results/step6
# stage 2 after Step 5 and after plan v1.0 + Amendment 1 is archived (OSF or Zenodo; Amendment 1, item 6).
# Repeats stage 1 (same numbers), adds test 2 (option B window: z over 2018-2020), family B, the premise check and RQ3.
python step6_link_analysis.py --signed-off-on 2026-10-11 --hindcast results/step2/hindcast_station_month_lstm5seeds.csv \
    --uncertainty results/step1c/station_monthly_l4_uncertainty.csv --step5 results/step5_final \
    --plan-archive <DOI or URL of the archived plan> --out results/step6
# exploratory checks of test 1 (not in the plan)
python step6d_exploratory_test1_inference.py --out results/step6/exploratory   # why permutation p and bootstrap interval differ
python step6e_exploratory_u_definition.py --out results/step6/exploratory      # alternative definitions of u
python step6f_exploratory_skill_bias.py --out results/step6/exploratory       # skill score with the mean bias removed
python step6b_exploratory_test1.py --hindcast results/step2/hindcast_station_month_lstm5seeds.csv \
    --uncertainty results/step1c/station_monthly_l4_uncertainty.csv --out results/step6/exploratory

# iQuam scoping for the L3 window (downloads and deletes each monthly file)
python iquam_scope_auto.py --out out/step2_iquam

# Step 2: forecasts at the 7 stations (input: repo data folder or the Step 1b CSV)
python step2_rolling_hindcast.py --input ../sst-lstm-bangladesh/data --replicate-ms1 --out out/step2
python step2_rolling_hindcast.py --input ../sst-lstm-bangladesh/data --seeds 42 43 44 45 46 --out out/step2

# Step 3: grid extension over 18-23 N, 86-93 E at 0.25 deg (persistence, climatology, SARIMA)
python step3_grid_hindcast.py --workers 4 --out out/step3          # best run on Google Colab
```

## Status (results in `results/`)

* Step 2 replication of Manuscript 1, Table 5: persistence, climatology and LSTM identical at
  all 7 stations; SARIMA identical at 6, Cox's Bazar 0.471 vs 0.474 (statsmodels version).
* Step 2 rolling hindcast 2015-2024, seed 42: done (840 station-months).
* Step 1b (run by the author on Colab, 1 Oct 2026): station SST reproduces Manuscript 1, Table 3
  (mean differences 0.000-0.004 degC, SD identical). The C3S dataset returned no uncertainty field.
* Step 1c, first run: the C3S dataset does carry an uncertainty field, named analysis_error_sst
  (1982-2024); the first detection rule missed the name. The ESACCI dataset (1980-2021) is a
  different analysis (monthly differences up to 1.26 degC), so its uncertainty is not used.
  Kept in results/step1c_esacci_check/ as a supporting check.
* Step 1c, second run (done): analysis_error_sst from the C3S dataset, 2000-2024; SST identical to
  the forecasting series. On 30 March of every year the field holds values near -273 K at all
  stations (175 station-days); these are set to missing. Cleaned monthly file: results/step1c/.
* Step 2 (done, run by the author on Colab): iQuam 2018-2020, results/step2_iquam/. Nine drifters,
  none within 50 km of a station (closest 68 km). Accuracy moves to the offshore NBoB.
* Step 3: analysis plan v1.0 (Claude doc) reviewed and corrected on 10 Oct 2026; approved by the
  supervisor without changes on 11 Oct 2026 (run log in the plan).
* Step 4 (done, run by the author on Colab, 12 Oct 2026, package 2026-10-12a): MetOp-B 2,189 files (4 days
  with one of the two files; one boundary file dated 2021-01-01), Aqua and Terra 2,192 each, 0 failed after retries,
  sst_dtime present in all. Station-box share of valid QL>=4 pixels (land included): monsoon 0.06-0.07, winter 0.39-0.46.
  Summary in results/step4/.
* Step 4 (user, Colab): step4_l3_download.py fetches 2018-2020 L3C subsets (18-23 N, 86-93 E)
  for MetOp-B AVHRR (OSI SAF, PO.DAAC, Earthdata login) and MODIS Aqua and Terra (NOAA ACSPO,
  CoastWatch THREDDS), then writes an inventory.
* Step 5: step5_reliability.py computes Accuracy (offshore drifter matchups), Availability and C1
  at the stations, C2 regionally, then screening and Li et al. scoring with sensitivity checks.
  Tested on one synthetic year (recovers injected biases and cloud fractions). Reads no forecast
  error.
* LSTM seeds 42-46 for the 2015-2024 hindcast (done 11 Oct 2026):
  results/step2/hindcast_station_month_lstm5seeds.csv. Mean RMSE: SARIMA 0.459, climatology 0.490,
  LSTM 0.503, persistence 1.827 degC. SARIMA lowest at all 7 stations; LSTM below climatology at 2.
  SARIMA forecasts differ from the first run by up to 0.10 degC in 3 station-years (statsmodels 0.15),
  mean RMSE unchanged at 0.459. Use the lstm5seeds file from now on.
* Step 6: step6_link_analysis.py implements Section 4 of analysis plan v1.0. It refuses real data
  without --signed-off-on DATE. Self-test: --synthetic. Power of test 1 with the real u series:
  results/power_test1.csv and power_test1_NOTE.txt.
* Step 6 stage 1 (done 11 Oct 2026, results/step6/NOTE.txt): primary test 1, SARIMA,
  beta = -0.35 z per 0.1 K (expected sign: positive), Freedman-Lane p = 0.042, 95% wild cluster
  bootstrap interval -0.97 to 0.26. No secondary variant survives FDR (smallest q = 0.33).
  Exploratory checks (step6b): whole-year permutation p = 0.037; leave-one-year-out beta -0.46 to -0.19.
  Test 2, family B and the premise check wait for Step 5.
  Further exploratory checks (11 Oct 2026, after a review): leave-one-station-out beta -0.37 to -0.30
  (results/step6/exploratory/loso_sarima.csv). u is the station-month mean of daily analysis_error_sst
  for each year (28-31 days per month after removing the 30 March values), as the plan defines it.
* Exploratory check of test 1 (12 Oct 2026, not in the plan; step6c_exploratory_anomaly.py,
  results/step6/exploratory/anomaly_check.csv): with the test 1 fixed effects and trend, the absolute L4 anomaly is
  unrelated to u (0.01 degC per 0.1 K, whole-year permutation p = 0.86, 2015-2024), so the damping explanation is not
  supported; the month-to-month anomaly change is slightly smaller in high-u months (-0.06 degC per 0.1 K, p = 0.41).
* Amendment 1 (proposed 11 Oct 2026, approved by the supervisor 13 Oct 2026, before Step 5 is run): Step 5 adds
  accuracy_regional_ci.csv (95% intervals from a bootstrap over drifter passes, leave-one-drifter-out
  ranges, share of the largest drifter) and availability_station_grid005.csv (all products on a common
  0.05 deg grid; a cell counts when at least half of its pixels are valid), with the ranking check
  "availability_common_grid_0.05deg" in sensitivity.csv. Primary tests and thresholds are unchanged.
  Item 7 (Test 2 window): first option A; changed on 13 Oct 2026, after the second external review and before any
  Test 2 statistic, to option B: mean z over 2018-2020 primary, 2015-2024 secondary (family B).
* Step 5 (done, run by the author on Colab, 13 Oct 2026, package 2026-10-12a; Step 5 code unchanged in 2026-10-13a):
  3,292 matchups from 9 drifters (MetOp-B 1,076, Aqua 1,060, Terra 1,156). Bathymetry: GEBCO_2023 was unreachable,
  so the built-in ETOPO1 fallback was used, as in the scoping (Deviation 1 in the plan). Offshore units with
  n < 50: monsoon day (all three products) and monsoon night (Aqua 33, Terra 44); MetOp-B monsoon night has 57.
  ACSPO MODIS L3C holds only quality level 5 pixels, so the QL 5 check changes MetOp-B only (B1 0.34 -> 0.11).
* Version 2026-10-13b (13 Oct 2026): rank_units() factored out of step5_reliability.main() (rerun of the ranking from
  the Colab tables reproduces recommendations, shares, sensitivity and screening exactly); step4 --months filter;
  step5b_accuracy_fallback.py implements the accuracy fallback of plan Section 3 for the units with n < 50.
* Step 6 stage 2 code (13 Oct 2026, version 2026-10-13a, written before any Step 5 output existed):
  family B now also covers MetOp-B B2, B3 and C1 and the 10 km and QL 5 runs (check_r10/, check_ql5/);
  premise_check.csv (MetOp-B monthly coverage vs u, 252 station-months); RQ3 as specified in the plan on
  13 Oct 2026: rq3_cells.csv, rq3_phase.csv, family C in link_secondary.csv (one factor at a time),
  rq3_joint.csv (mean SARIMA z ~ phase + sigma + amplitude + effective-day share, exact station-block
  permutation, wild cluster bootstrap by station) and confidence_table.csv. Stage 1 numbers are unchanged
  (rerun agrees to 2e-14). Stage 2 refuses to run without --plan-archive.

## Outputs to check

| Step | File | Decision it supports |
|---|---|---|
| 1a | `table1_region_phase_daynight.csv`, `fig_drifter_counts.png` | Do shelf / offshore x phase x day/night units reach n >= 50 (Li et al.)? Sets the Part A unit and window. |
| 1a | `table2_station_radius_phase.csv` | Confirms whether station-scale Accuracy is possible (expected: no). |
| 1b | `replication_check_ms1_table3.csv` | Extraction reproduces Manuscript 1 (mean diff about 0.00). |
| 1b | `station_monthly_l4_uncertainty.csv` | Main data-support predictor for Part C. |
| 2 | `ms1_replication_check.csv` | Pipeline reproduces MS1 Table 5 before any new result is trusted. |
| 2 | `hindcast_station_month.csv` | 840 station-months x 4 methods, input for Part C. |

## Design decisions fixed so far

* Monsoon phases: pre-monsoon MAM, monsoon JJAS, post-monsoon ON, winter DJF.
* Day/night by solar zenith angle: day < 85 deg, night > 95 deg, twilight excluded (Li et al.).
* In situ reference: iQuam quality_level >= 4; drifters are the primary reference, other
  platforms are counted separately and used only if the design document says so.
* Forecast errors: rolling origin, one test year per fold, 2015-2024, training always
  ends in December of the previous year. Errors are later standardised by
  `sd_train_calmonth`, computed from training years only.
* Scoping counts are in situ observations, an upper bound on satellite matchups.

* Version 2026-10-13c (13 Oct 2026, second external review): step6 primary test 2 uses z over 2018-2020 (option B);
  family B variant B1_metopb_errors_2015_2024; RQ3 joint model also with z over 2018-2020 (rq3_joint_z2018_2020.csv).
  Stage 1 rerun reproduces link_primary.csv and link_secondary.csv exactly.
  Exploratory checks (results/step6/exploratory/): test1_inference_check.csv, test1_years.csv,
  test1_inverted_pvalues.csv (step6d) and test1_u_definitions.csv (step6e). Permutation with the classical t: p 0.048;
  with the cluster-robust t: 0.18; restricted wild cluster bootstrap: 0.21 (Webb), 0.22 (Rademacher, exact);
  inverted permutation interval -0.66 to -0.02 (classical t), -0.88 to 0.18 (cluster-robust t) per 0.1 K.
  Alternative u definitions: beta -0.08 to -0.09 z per within-SD of u in every case.

* Version 2026-10-13d (13 Oct 2026, Step 5b): the 2016-2017 extension (June-September) found no drifter in the regional
  box (1,506 iQuam records from one moored buoy, 12 Argo floats and six ships; results/step1a_ext, results/step5_ext),
  so the five monsoon units below 50 matchups stay descriptive only (plan Section 3). Final RQ1 tables:
  results/step5_final (Step 6 reads --step5 results/step5_final). step5b labels a unit as extended only when matchups
  were added, and recomputes the accuracy intervals for all units from the saved matchups: this reproduces the Step 5
  bootstrap intervals and corrects the drifter counts of the winter offshore units (5 -> 4) and three
  leave-one-drifter-out ranges, because the Colab run had read one drifter id once as a number and once as text.
  step5_reliability.read_iquam now converts platform ids to one text form. Scores and rankings are unchanged.

* Version 2026-10-13e (13 Oct 2026, Step 6 stage 2, Deviation 2): stage 2 was run before the public archive of the
  plan, at the student's decision, with --archive-deviation in place of --plan-archive (the plan PDF 2026-10-13c,
  SHA-256 08f1d348fcf92a30a5fe98c879dd25fbf74162420c054c65742db0a98ebad04e, was frozen first). Command:
    python step6_link_analysis.py --signed-off-on 2026-10-11 \
      --hindcast results/step2/hindcast_station_month_lstm5seeds.csv \
      --uncertainty results/step1c/station_monthly_l4_uncertainty.csv --step5 results/step5_final \
      --archive-deviation "Deviation 2, plan run log 13 Oct 2026; plan PDF 2026-10-13c SHA-256 08f1d348..." \
      --out results/step6_stage2
  Stage 1 numbers are reproduced exactly. Test 2: rho -0.12, exact p 0.42. Outputs in results/step6_stage2.

* Version 2026-10-13f (13 Oct 2026, peer review of the manuscript): step6f_exploratory_skill_bias.py (not in the plan)
  splits the MSE of SARIMA and climatology by phase into squared mean bias and remainder
  (results/step6/exploratory/skill_bias_by_phase.csv, skill_bias_monsoon_by_station.csv). In the monsoon the training
  climatology was 0.22 C too cold (0.19-0.26 C by station) and SARIMA 0.11 C; with the mean bias removed, the SARIMA
  MSSS is -0.08 in the monsoon (-0.22 to +0.01 by station), 0.17 pre-monsoon, 0.14 post-monsoon, -0.01 winter.
  No other result changes.
  requirements.txt now lists scipy, which step6 uses.
