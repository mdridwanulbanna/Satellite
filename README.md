# SST reliability and forecast skill in the northern Bay of Bengal

Analysis code, frozen analysis plan and run log for two manuscripts on satellite sea surface temperature (SST) along the Bangladesh coast:

1. **Monsoon-Stratified and Diurnal Reliability of Level-3 Satellite Sea Surface Temperature Products in the Northern Bay of Bengal** (accuracy, availability and consistency of three Level-3 products, 2018–2020).
2. **Multi-Model Hindcast Skill of Coastal Sea Surface Temperature Along the Bangladesh Coast and Its Association with Satellite Data Support** (four one-month-ahead methods, rolling origin 2015–2024, and two association tests).

Authors: Md. Ridwanul Banna Rifat, Mohammad Muslem Uddin, Nishat Anjum Farzin. Department of Oceanography, Faculty of Marine Sciences and Fisheries, University of Chittagong, Chattogram 4331, Bangladesh.

**Archive:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

## What is in this repository

| Step | Script | Purpose |
| --- | --- | --- |
| 1a | `step1a_iquam_scoping.py`, `iquam_scope_auto.py` | iQuam drifter scoping, 2018–2020 |
| 1b | `step1b_l4_station_series.py` | Monthly L4 SST at the seven station cells, with a reproduction check |
| 1c | `step1c_l4_uncertainty.py` | L4 analysis uncertainty `u`, 2000–2024 |
| 2 | `step2_rolling_hindcast.py` | Rolling-origin hindcast of four methods (persistence, climatology, SARIMA, five-seed LSTM), 2015–2024 |
| 3 | `step3_grid_hindcast.py` | Grid extension of the hindcast over the regional box. Not run and not used in the primary tests |
| 4 | `step4_l3_download.py` | L3 subsets for MetOp-B, MODIS Aqua and MODIS Terra, 2018–2020 |
| 5 | `step5_reliability.py` | Matchups, metrics A1 to C2, screening, scores and sensitivity checks (manuscript 1) |
| 5b | `step5b_accuracy_fallback.py` | Accuracy fallback: 2016–2017 monsoon matchups for units with fewer than 50 matchups |
| 6 | `step6_link_analysis.py` | Test 1, Test 2, premise check and secondary families (manuscript 2) |
| 6b–6f | `step6b_exploratory_test1.py`, `step6c_exploratory_anomaly.py`, `step6d_exploratory_test1_inference.py`, `step6e_exploratory_u_definition.py`, `step6f_exploratory_skill_bias.py` | Exploratory checks that accompany the tests |

The frozen analysis plan (PDF), its run log and the SHA-256 checksum of the plan are archived with the code.

## Data

The data are public and are not redistributed here. Download them from the providers.

| Data | Provider | Identifier |
| --- | --- | --- |
| MetOp-B AVHRR L3C SST (GHRSST GDS V2), 0.05°, day and night | EUMETSAT OSI SAF via NASA PO.DAAC | https://doi.org/10.5067/GHGMB-3CO02 |
| ACSPO MODIS Aqua and Terra L3C SST, 0.02° | NOAA STAR via NOAA CoastWatch | https://coastwatch.noaa.gov/ (product description: https://doi.org/10.3390/rs15235589) |
| In situ SST Quality Monitor (iQuam) v2.10, drifting buoys | NOAA STAR | https://www.star.nesdis.noaa.gov/sod/sst/iquam/ (method: https://doi.org/10.1175/JTECH-D-13-00121.1) |
| ESA SST CCI and C3S reprocessed L4 SST analyses, with `analysis_error_sst` | Copernicus Marine Service | https://doi.org/10.48670/moi-00169 (`SST_GLO_SST_L4_REP_OBSERVATIONS_010_024`) |
| ETOPO1 1 arc-minute relief model | NOAA National Geophysical Data Center | https://doi.org/10.7289/V5C8276M |

GEBCO_2023 was planned for the bathymetry but its server could not be reached, so the code uses the ETOPO1 fallback. This is documented as Deviation 1 in the run log.

## Requirements

Python 3 with xarray, pandas, NumPy, SciPy, statsmodels, TensorFlow and Matplotlib. Pin the exact versions in `requirements.txt` before archiving. The SARIMA fits use the `SARIMAX` routine of statsmodels, so results can differ slightly between statsmodels versions.

## How to run

1. Install the requirements and download the data listed above.
2. Set the local data and output folders at the top of each script.
3. Run the scripts in the order of the table above. Steps 1a to 2 and 4 to 5b feed manuscripts 1 and 2. Step 6 needs the outputs of steps 1c, 2 and 5.

## Reproducibility notes

- Code version label: `2026-10-13f`.
- Random seeds: 20261011 for the link tests and 20261013 for the exploratory checks. The LSTM uses five seeds, 42 to 46, and the five forecasts are averaged.
- The analysis plan was approved by the supervisor on 11 October 2026 (plan v1.0) and amended on 13 October 2026 (Amendment 1). It is prespecified by supervisor approval and is not publicly preregistered. The run log records the order of the later steps, including Deviation 2 (Test 2, the premise check, Families B and C and the exploratory research question were run before this archive was prepared).
- The L4 series used here reproduced the station means of the earlier station-forecasting study within 0.004 °C.

## Citation

Citation details for the code are in CITATION.cff.

## License

MIT (See `LICENSE`).
