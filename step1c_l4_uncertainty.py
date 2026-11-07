"""Step 1c: locate the L4 analysis uncertainty and extract it at the seven stations.

Step 1b showed that the C3S dataset used for the forecasts returns SST but no
uncertainty variable. The same Copernicus Marine product also holds the ESA SST CCI
dataset (ESACCI-GLO-SST-L4-REP-OBS-SST), which normally carries
analysed_sst_uncertainty. This script:
  1. lists the variables and time range of both datasets,
  2. extracts SST and uncertainty from the first dataset that has an uncertainty field,
  3. checks that its monthly SST equals the C3S monthly SST used for the forecasts,
     so that its uncertainty can be attached to the forecast errors.

  python step1c_l4_uncertainty.py --c3s-monthly results/step1b/station_monthly_sst.csv --out out/step1c
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

import config as C
from step1b_l4_station_series import extract, find_uncertainty_var

DATASETS = ["C3S-GLO-SST-L4-REP-OBS-SST", "ESACCI-GLO-SST-L4-REP-OBS-SST"]


def open_box(cm, dsid, **kw):
    lats = [s[3] for s in C.STATIONS]
    lons = [s[4] for s in C.STATIONS]
    return cm.open_dataset(dataset_id=dsid,
                           minimum_latitude=min(lats) - 0.1, maximum_latitude=max(lats) + 0.1,
                           minimum_longitude=min(lons) - 0.1, maximum_longitude=max(lons) + 0.1, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c3s-monthly", default="results/step1b/station_monthly_sst.csv")
    ap.add_argument("--out", default="out/step1c")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    vfile = os.path.join(here, "VERSION")
    print("Package version:", open(vfile).read().strip() if os.path.exists(vfile) else "OLD (no VERSION file)")
    try:
        import copernicusmarine as cm
    except ImportError:
        sys.exit("pip install copernicusmarine")

    # 1. what does each dataset contain?
    chosen, info = None, []
    for dsid in DATASETS:
        try:
            ds = open_box(cm, dsid)
            t = pd.to_datetime(ds["time"].values)
            unc = find_uncertainty_var(ds)
            info.append(dict(dataset=dsid, variables=", ".join(ds.data_vars),
                             first_day=t.min().date(), last_day=t.max().date(), uncertainty=unc))
            if unc and chosen is None:
                chosen = (dsid, t.min(), t.max())
        except Exception as e:
            info.append(dict(dataset=dsid, variables=f"ERROR: {e}", first_day=None, last_day=None,
                             uncertainty=None))
    info = pd.DataFrame(info)
    info.to_csv(os.path.join(a.out, "dataset_inventory.csv"), index=False)
    print("\n=== Datasets in SST_GLO_SST_L4_REP_OBSERVATIONS_010_024 ===")
    print(info.to_string(index=False))
    if chosen is None:
        sys.exit("\nNo dataset with an uncertainty field. Send this printout in the chat.")

    dsid, t0, t1 = chosen
    start = max(pd.Timestamp("2000-01-01"), t0).strftime("%Y-%m-%d")
    end = min(pd.Timestamp("2024-12-31"), t1).strftime("%Y-%m-%d")
    print(f"\nUsing {dsid} for the uncertainty, {start} to {end}")

    # 2. extract SST and uncertainty at the Manuscript 1 grid cells
    ds = open_box(cm, dsid, start_datetime=start, end_datetime=end)
    daily, unc, mask_info = extract(ds, start, end)
    daily.to_csv(os.path.join(a.out, "station_daily_l4_uncertainty.csv"), index=False)
    if mask_info:
        print("\nMask values at each grid cell (value: days):")
        for m in mask_info:
            print(f"  {m['station']}: {m['mask_values']}")

    daily["date_m"] = daily["date"].dt.to_period("M").dt.to_timestamp()
    g = daily.groupby(["station", "date_m"])
    monthly = pd.DataFrame({
        "sst_c": g["sst_c"].mean(),
        "unc_mean_k": g["unc_k"].mean(),
        "unc_rms_k": g["unc_k"].apply(lambda s: float(np.sqrt(np.nanmean(s ** 2)))),
        "n_days": g["unc_k"].count(),
    }).reset_index().rename(columns={"date_m": "date"})
    monthly["phase"] = monthly["date"].dt.month.map(C.PHASE_OF_MONTH)
    monthly["source_dataset"] = dsid
    monthly.to_csv(os.path.join(a.out, "station_monthly_l4_uncertainty.csv"), index=False)

    # 3. is this the same analysis as the C3S series used for the forecasts?
    if os.path.exists(a.c3s_monthly):
        c3s = pd.read_csv(a.c3s_monthly, parse_dates=["date"]).melt(
            id_vars="date", var_name="station", value_name="sst_c3s")
        m = monthly.merge(c3s, on=["station", "date"])
        m["diff"] = m["sst_c"] - m["sst_c3s"]
        chk = m.groupby("station")["diff"].agg(
            n_months="count", mean_diff="mean", max_abs_diff=lambda d: d.abs().max()).round(4)
        chk.to_csv(os.path.join(a.out, "consistency_vs_c3s.csv"))
        print("\n=== Monthly SST of this dataset minus the C3S series used for forecasting ===")
        print(chk.to_string())
        worst = float(chk["max_abs_diff"].max())
        print("\nVERDICT:", "same analysis, uncertainty can be attached to the forecast errors"
              if worst < 0.05 else f"series differ (max {worst:.3f} degC): send this printout before going on")
    else:
        print(f"\n{a.c3s_monthly} not found; consistency check skipped.")

    print("\n=== Mean daily L4 uncertainty (K) by station and phase ===")
    print(monthly.pivot_table(index="station", columns="phase", values="unc_mean_k")
          [C.PHASE_ORDER].round(3).to_string())
    print(f"\nOutputs written to {a.out}")


if __name__ == "__main__":
    main()
