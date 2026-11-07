"""Step 3: extend the forecast-skill analysis from 7 stations to a regular grid over the NBoB.

Why: with 7 stations the spatial part of the analysis has 7 independent units. A 0.25 deg
grid over 18-23 N, 86-93 E gives several hundred ocean points covering shelf, slope and
open bay, so the link between L4 observational support and forecast error can be tested
across space as well as across years.

What it does
  1. Reads daily C3S L4 analysed_sst and analysed_sst_uncertainty (online or local file),
     keeps every `stride`-th cell (stride 5 on the 0.05 deg grid = 0.25 deg), and drops
     cells that are not ice-free ocean on every day.
  2. Writes monthly means of SST and uncertainty to grid_monthly.nc (reused later).
  3. Runs the rolling-origin hindcast (default 2015-2024) with persistence, climatology and
     SARIMA at every grid point, exactly as Step 2 does for the stations. The LSTM is not
     run on the grid by default (70 fits per point); Step 2 covers it at the 7 stations.
  4. Writes grid_hindcast_point_month.csv; the run can be stopped and resumed.

  python step3_grid_hindcast.py --local-file nbob_l4_daily.nc --workers 4 --out out/step3
  python step3_grid_hindcast.py --workers 4 --out out/step3            # online (copernicusmarine)
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import xarray as xr

import config as C
from step1b_l4_station_series import coord_name, find_uncertainty_var
from step2_rolling_hindcast import (climatology_forecasts, persistence_forecasts, sarima_forecasts,
                                    sd_train_calmonth)

BOX = dict(lat_min=18.0, lat_max=23.0, lon_min=86.0, lon_max=93.0)


def build_monthly_grid(args):
    path = os.path.join(args.out, "grid_monthly.nc")
    if os.path.exists(path):
        print("Using existing", path)
        return xr.open_dataset(path)
    if args.local_file:
        ds = xr.open_dataset(args.local_file)
    else:
        import copernicusmarine as cm
        ds = cm.open_dataset(dataset_id=C.L4_DATASET_ID,
                             minimum_latitude=BOX["lat_min"], maximum_latitude=BOX["lat_max"],
                             minimum_longitude=BOX["lon_min"], maximum_longitude=BOX["lon_max"],
                             start_datetime="2000-01-01", end_datetime="2024-12-31")
    latn, lonn = coord_name(ds, ["latitude", "lat"]), coord_name(ds, ["longitude", "lon"])
    ds = ds.sortby([latn, lonn])
    ds = ds.sel({latn: slice(BOX["lat_min"], BOX["lat_max"]), lonn: slice(BOX["lon_min"], BOX["lon_max"]),
                 "time": slice("2000-01-01", "2024-12-31")})
    ds = ds.isel({latn: slice(0, None, args.stride), lonn: slice(0, None, args.stride)})
    unc = find_uncertainty_var(ds)
    if unc is None:
        sys.exit("analysed_sst uncertainty variable not found; cannot build the grid predictor.")
    keep = ["analysed_sst", unc] + (["mask"] if "mask" in ds.data_vars else [])
    ds = ds[keep].load()
    sst = ds["analysed_sst"]
    if float(sst.mean()) > 200:
        sst = sst - 273.15
    ocean = sst.notnull().all("time")
    if "mask" in ds:
        ocean = ocean & (ds["mask"] == 1).all("time")
    out = xr.Dataset({
        "sst_c": sst.resample(time="MS").mean(),
        "unc_mean_k": ds[unc].resample(time="MS").mean(),
        "unc_rms_k": np.sqrt((ds[unc] ** 2).resample(time="MS").mean()),
        "ocean": ocean,
    }).rename({latn: "lat", lonn: "lon"})
    out.to_netcdf(path)
    print(f"Wrote {path}: {int(out['ocean'].sum())} ocean points")
    return out


def run_point(task):
    pid, lat, lon, values, dates, first, last = task
    series = pd.Series(values, index=pd.DatetimeIndex(dates))
    frames = []
    for year in range(first, last + 1):
        train_end = pd.Timestamp(f"{year - 1}-12-01")
        idx = pd.date_range(f"{year}-01-01", f"{year}-12-01", freq="MS")
        sar, conv = sarima_forecasts(series, train_end, idx)
        frames.append(pd.DataFrame({
            "point_id": pid, "lat": lat, "lon": lon, "fold": year, "date": idx,
            "month": idx.month, "phase": idx.month.map(C.PHASE_OF_MONTH),
            "obs": series.loc[idx].values,
            "pred_persistence": persistence_forecasts(series, idx),
            "pred_climatology": climatology_forecasts(series, train_end, idx),
            "pred_sarima": sar, "sarima_converged": conv,
            "sd_train_calmonth": sd_train_calmonth(series, train_end, idx),
        }))
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-file", default=None)
    ap.add_argument("--stride", type=int, default=5, help="5 x 0.05 deg = 0.25 deg")
    ap.add_argument("--first-year", type=int, default=2015)
    ap.add_argument("--last-year", type=int, default=2024)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-points", type=int, default=None, help="for quick tests")
    ap.add_argument("--out", default="out/step3")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    g = build_monthly_grid(a)
    pts = g["ocean"].to_dataframe().reset_index()
    pts = pts[pts["ocean"]].reset_index(drop=True)
    pts["point_id"] = [f"p{i:04d}" for i in range(len(pts))]
    pts.to_csv(os.path.join(a.out, "grid_points.csv"), index=False)
    # monthly uncertainty in long format, for the analysis step
    unc = g[["unc_mean_k", "unc_rms_k"]].to_dataframe().reset_index()
    unc = unc.merge(pts[["lat", "lon", "point_id"]], on=["lat", "lon"])
    unc.rename(columns={"time": "date"}).to_csv(os.path.join(a.out, "grid_monthly_uncertainty.csv"), index=False)

    if a.max_points:
        pts = pts.head(a.max_points)

    result_path = os.path.join(a.out, "grid_hindcast_point_month.csv")
    done = set(pd.read_csv(result_path, usecols=["point_id"])["point_id"]) if os.path.exists(result_path) else set()
    dates = g["time"].values
    tasks = [(r.point_id, float(r.lat), float(r.lon),
              g["sst_c"].sel(lat=r.lat, lon=r.lon).values.astype(float), dates, a.first_year, a.last_year)
             for r in pts.itertuples() if r.point_id not in done]
    print(f"{len(pts)} ocean points, {len(done)} already done, {len(tasks)} to run")

    t0 = time.time()
    with Pool(a.workers) as pool:
        for k, df in enumerate(pool.imap_unordered(run_point, tasks), 1):
            df.to_csv(result_path, mode="a", header=not os.path.exists(result_path), index=False)
            if k % 10 == 0 or k == len(tasks):
                rate = (time.time() - t0) / k
                print(f"{k}/{len(tasks)} points, {rate:.1f} s/point, ~{rate * (len(tasks) - k) / 60:.0f} min left")
    print("Done:", result_path)


if __name__ == "__main__":
    main()
