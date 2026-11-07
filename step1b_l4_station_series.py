"""Step 1b: station time series of C3S L4 SST and its analysis uncertainty.

Extracts daily analysed_sst and analysed_sst_uncertainty (plus mask, if present) at the
seven grid cells of Manuscript 1, Table 1, for 2000-2024, then writes
  station_daily_l4.csv                 daily values, long format
  station_monthly_sst.csv              monthly mean SST (degC), wide format -> input of Step 2
  station_monthly_l4_uncertainty.csv   monthly mean and RMS of the daily uncertainty (K = degC)
  replication_check_ms1_table3.csv     mean/SD vs Manuscript 1 Table 3

Two ways to get the data:
  (A) online, via the Copernicus Marine toolbox (run `copernicusmarine login` once):
      python step1b_l4_station_series.py --out out/step1b
  (B) from a netCDF file you already downloaded for the box (must contain the variables):
      python step1b_l4_station_series.py --local-file my_subset.nc --out out/step1b

Note on uncertainty: daily analysis errors are correlated in time, so the monthly mean
uncertainty is NOT reduced by sqrt(days). We report the plain mean and the RMS of the
daily values and treat both as indices of observational support, not as the error of
the monthly mean.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

import config as C


def coord_name(ds, options):
    for o in options:
        if o in ds.coords or o in ds.variables:
            return o
    raise KeyError(f"None of {options} in dataset")


def find_uncertainty_var(ds):
    """SST analysis uncertainty: 'analysis_error_sst' in the C3S dataset,
    'analysed_sst_uncertainty' in the ESACCI dataset. Ice-surface fields (..._st) are ignored."""
    names = list(ds.data_vars)
    for v in names:
        if str(v).lower() in ("analysis_error_sst", "analysed_sst_uncertainty", "analysed_sst_error"):
            return v
    for v in names:
        lv = str(v).lower()
        if ("uncert" in lv or "error" in lv) and "sst" in lv:
            return v
    return None


def open_online(start, end):
    try:
        import copernicusmarine as cm
    except ImportError:
        sys.exit("pip install copernicusmarine   (then run: copernicusmarine login)")
    lats = [s[3] for s in C.STATIONS]
    lons = [s[4] for s in C.STATIONS]
    box = dict(minimum_latitude=min(lats) - 0.1, maximum_latitude=max(lats) + 0.1,
               minimum_longitude=min(lons) - 0.1, maximum_longitude=max(lons) + 0.1,
               start_datetime=start, end_datetime=end)
    try:  # ask only for the variables we need; fall back to all variables if a name differs
        ds = cm.open_dataset(dataset_id=C.L4_DATASET_ID,
                             variables=["analysed_sst", "analysis_error_sst", "mask"], **box)
    except Exception as e:
        print("Variable-specific request failed (", e, "); opening all variables instead")
        ds = cm.open_dataset(dataset_id=C.L4_DATASET_ID, **box)
    return ds


def extract(ds, start, end):
    latn = coord_name(ds, ["latitude", "lat"])
    lonn = coord_name(ds, ["longitude", "lon"])
    timen = coord_name(ds, ["time"])
    ds = ds.sel({timen: slice(start, end)})
    unc = find_uncertainty_var(ds)
    print("Variables in dataset:", list(ds.data_vars))
    print("Uncertainty variable:", unc if unc else "NOT FOUND (Step 1b cannot provide it)")
    wanted = ["analysed_sst"] + ([unc] if unc else []) + (["mask"] if "mask" in ds.data_vars else [])

    frames, mask_info = [], []
    for name, _, _, glat, glon in C.STATIONS:
        pt = ds[wanted].sel({latn: glat, lonn: glon}, method="nearest")
        got_lat, got_lon = float(pt[latn]), float(pt[lonn])
        if abs(got_lat - glat) > 0.01 or abs(got_lon - glon) > 0.01:
            print(f"WARNING {name}: nearest cell ({got_lat:.3f}, {got_lon:.3f}) "
                  f"differs from Manuscript 1 cell ({glat}, {glon})")
        pt = pt.load()
        df = pd.DataFrame({"date": pd.to_datetime(pt[timen].values)})
        sst = pt["analysed_sst"].values.astype(float)
        if np.nanmean(sst) > 200:
            sst = sst - 273.15  # K -> degC for the temperature itself only
        df["sst_c"] = sst
        if unc:
            u = pt[unc].values.astype(float)  # a difference: no 273.15 offset
            bad = (u < 0) | (u > 10)
            if bad.any():
                print(f"  {name}: {int(bad.sum())} days with invalid uncertainty "
                      f"(e.g. {u[bad][0]:.2f} K) set to missing")
            df["unc_k"] = np.where(bad, np.nan, u)
        if "mask" in wanted:
            vals, counts = np.unique(pt["mask"].values, return_counts=True)
            mask_info.append(dict(station=name, mask_values=dict(zip(vals.tolist(), counts.tolist()))))
        df["station"] = name
        df["grid_lat"], df["grid_lon"] = got_lat, got_lon
        frames.append(df)
    return pd.concat(frames, ignore_index=True), unc, mask_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-file", default=None)
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--out", default="out/step1b")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ds = xr.open_dataset(a.local_file) if a.local_file else open_online(a.start, a.end)
    daily, unc, mask_info = extract(ds, a.start, a.end)
    daily.to_csv(os.path.join(a.out, "station_daily_l4.csv"), index=False)

    n_missing = daily.groupby("station")["sst_c"].apply(lambda s: s.isna().sum())
    print("\nMissing daily SST values per station:\n", n_missing.to_string())
    if mask_info:
        print("\nMask values at each grid cell (value: days):")
        for m in mask_info:
            print(f"  {m['station']}: {m['mask_values']}")

    daily["month_start"] = daily["date"].dt.to_period("M").dt.to_timestamp()
    g = daily.groupby(["station", "month_start"])
    monthly_sst = g["sst_c"].mean().unstack("station")[C.STATION_NAMES]
    monthly_sst.index.name = "date"
    monthly_sst.round(4).to_csv(os.path.join(a.out, "station_monthly_sst.csv"))

    if unc:
        mu = g["unc_k"].mean().rename("unc_mean_k")
        rms = g["unc_k"].apply(lambda s: float(np.sqrt(np.nanmean(s ** 2)))).rename("unc_rms_k")
        out = pd.concat([mu, rms], axis=1).reset_index().rename(columns={"month_start": "date"})
        out["phase"] = out["date"].dt.month.map(C.PHASE_OF_MONTH)
        out.to_csv(os.path.join(a.out, "station_monthly_l4_uncertainty.csv"), index=False)
        print("\nMean daily L4 uncertainty (K) by station and phase:")
        print(out.pivot_table(index="station", columns="phase", values="unc_mean_k")
              [C.PHASE_ORDER].round(3).to_string())

    # replication check against Manuscript 1, Table 3
    rows = []
    for st in C.STATION_NAMES:
        s = monthly_sst[st]
        ref_mean, ref_sd = C.MS1_TABLE3[st]
        rows.append(dict(station=st, mean=round(s.mean(), 2), ms1_mean=ref_mean,
                         sd=round(s.std(ddof=1), 2), ms1_sd=ref_sd,
                         mean_diff=round(s.mean() - ref_mean, 3)))
    chk = pd.DataFrame(rows)
    chk.to_csv(os.path.join(a.out, "replication_check_ms1_table3.csv"), index=False)
    print("\nReplication check vs Manuscript 1 Table 3 (diffs should be ~0.00-0.01):")
    print(chk.to_string(index=False))
    print(f"\nOutputs written to {a.out}")


if __name__ == "__main__":
    main()
