"""Step 1a: how much in-situ reference SST exists in the northern Bay of Bengal?

Reads NOAA iQuam v2.10 monthly netCDF files, keeps records inside SCOPING_BOX with
quality_level >= 4, and counts them by
  (a) sub-region (shelf < 200 m / offshore >= 200 m, from GEBCO) x monsoon phase x day/night
  (b) distance to each of the seven stations (5, 10, 25, 50 km) x monsoon phase.

The counts are an UPPER BOUND on satellite matchups: a matchup also needs a clear-sky,
quality_level >= 4 satellite pixel within +/-3 h and 5 km. Compare them with the Li et al.
thresholds (n >= 50 per unit for Accuracy, >= 30 per shelf/deep class for C3).

Usage
  python step1a_iquam_scoping.py --inspect path/to/one_iquam_file.nc
  python step1a_iquam_scoping.py --iquam-dir data/iquam --gebco data/gebco_nbob.nc --out out/step1a
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

import config as C

RADII_KM = [5, 10, 25, 50]
# Fallback platform codes, used ONLY if the file carries no flag_meanings attribute.
# Verify against --inspect output before trusting them.
FALLBACK_PLATFORMS = {1: "ship", 2: "drifter", 3: "tropical_moored", 4: "coastal_moored",
                      5: "argo", 6: "drifter_hr", 7: "imos", 8: "crw_buoy"}


def find_var(ds, candidates, required=True):
    for name in candidates:
        if name in ds.variables:
            return name
    lower = {v.lower(): v for v in ds.variables}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    if required:
        raise KeyError(f"None of {candidates} found. Variables: {list(ds.variables)}")
    return None


def platform_lookup(da):
    vals = da.attrs.get("flag_values")
    meanings = da.attrs.get("flag_meanings")
    if vals is not None and meanings:
        names = str(meanings).split()
        return {int(v): n for v, n in zip(np.atleast_1d(vals), names)}, "file attributes"
    return FALLBACK_PLATFORMS, "FALLBACK (check with --inspect)"


def platform_group(name):
    n = name.lower()
    if "drift" in n:
        return "drifter"
    if "moor" in n or "crw" in n or "buoy" in n:
        return "moored"
    if "argo" in n:
        return "argo"
    if "ship" in n:
        return "ship"
    return "other"


def get_times(ds):
    """Return numpy datetime64[s] times whatever the file layout is."""
    tname = find_var(ds, ["time", "obs_time", "sst_time"], required=False)
    if tname is not None and np.issubdtype(ds[tname].dtype, np.datetime64):
        return ds[tname].values.astype("datetime64[s]")
    parts = [find_var(ds, [p], required=False) for p in ["year", "month", "day", "hour", "minute", "second"]]
    if all(p is not None for p in parts[:4]):
        y, mo, d, h = (ds[p].values.astype(int) for p in parts[:4])
        mi = ds[parts[4]].values.astype(int) if parts[4] else 0
        se = ds[parts[5]].values.astype(int) if parts[5] else 0
        df = pd.DataFrame(dict(year=y, month=mo, day=d, hour=h, minute=mi, second=se))
        return pd.to_datetime(df, errors="coerce").values.astype("datetime64[s]")
    raise KeyError("Could not decode observation time. Run --inspect and edit get_times().")


def decode_ids(arr):
    arr = np.asarray(arr)
    if arr.dtype.kind == "S" and arr.ndim == 2:
        arr = np.array([b"".join(r).strip() for r in arr])
    if arr.dtype.kind in ("S", "O"):
        return np.array([x.decode("ascii", "ignore").strip() if isinstance(x, bytes) else str(x).strip()
                         for x in arr])
    return arr.astype(str)


def inspect(path):
    ds = xr.open_dataset(path, decode_times=True)
    print(ds)
    for v in ds.variables:
        a = ds[v].attrs
        keep = {k: a[k] for k in a if k in ("units", "long_name", "flag_values", "flag_meanings")}
        print(f"  {v}: dtype={ds[v].dtype}, shape={ds[v].shape}, {keep}")


def read_one(path, box, ql_min):
    ds = xr.open_dataset(path, decode_times=True, decode_timedelta=False)
    lat = ds[find_var(ds, ["lat", "latitude"])].values.astype(float)
    lon = ds[find_var(ds, ["lon", "longitude"])].values.astype(float)
    lon = np.where(lon > 180, lon - 360, lon)
    inbox = ((lat >= box["lat_min"]) & (lat <= box["lat_max"]) &
             (lon >= box["lon_min"]) & (lon <= box["lon_max"]))
    if not inbox.any():
        return pd.DataFrame(), None
    ql = ds[find_var(ds, ["quality_level"])].values
    keep = inbox & (ql >= ql_min)
    idx = np.where(keep)[0]
    if idx.size == 0:
        return pd.DataFrame(), None
    sst = ds[find_var(ds, ["sst", "sea_surface_temperature"])].values[idx].astype(float)
    if np.nanmean(sst) > 200:  # kelvin
        sst = sst - 273.15
    pt_name = find_var(ds, ["platform_type"])
    lookup, source = platform_lookup(ds[pt_name])
    ptype = ds[pt_name].values[idx].astype(int)
    pid_name = find_var(ds, ["platform_id", "id", "wmo_id"], required=False)
    pid = decode_ids(ds[pid_name].values[idx]) if pid_name else np.full(idx.size, "unknown")
    times = get_times(ds)[idx]
    df = pd.DataFrame(dict(time=times, lat=lat[idx], lon=lon[idx], sst_c=sst,
                           quality_level=ql[idx].astype(int), platform_code=ptype, platform_id=pid))
    df["platform_name"] = df["platform_code"].map(lookup).fillna("unknown")
    df["platform_group"] = df["platform_name"].map(platform_group)
    return df, source


def add_depth(df, gebco_path):
    if not gebco_path:
        df["depth_m"] = np.nan
        df["subregion"] = "unclassified"
        return df
    g = xr.open_dataset(gebco_path)
    elev = g[find_var(g, ["elevation", "altitude", "z", "Band1"])]
    latn = find_var(g, ["lat", "latitude"])
    lonn = find_var(g, ["lon", "longitude"])
    e = elev.sel({latn: xr.DataArray(df["lat"].values, dims="p"),
                  lonn: xr.DataArray(df["lon"].values, dims="p")}, method="nearest").values
    df["depth_m"] = -e.astype(float)
    df["subregion"] = np.where(df["depth_m"] <= 0, "land_or_coast",
                               np.where(df["depth_m"] < C.SHELF_DEPTH_M, "shelf", "offshore"))
    return df


def summarise_and_save(obs, gebco, out, ql_min):
    """Tag phase, day/night, depth and station distances, then write the scoping tables."""
    a = argparse.Namespace(gebco=gebco, out=out, ql_min=ql_min)
    os.makedirs(out, exist_ok=True)
    obs["month"] = pd.DatetimeIndex(obs["time"]).month
    obs["year"] = pd.DatetimeIndex(obs["time"]).year
    obs["phase"] = obs["month"].map(C.PHASE_OF_MONTH)
    obs["sza"] = C.solar_zenith_deg(obs["time"].values, obs["lat"].values, obs["lon"].values)
    obs["daynight"] = C.day_night_label(obs["sza"])
    obs["date"] = pd.DatetimeIndex(obs["time"]).normalize()
    obs = add_depth(obs, a.gebco)

    # distance to every station
    for name, slat, slon, _, _ in C.STATIONS:
        obs[f"dist_km::{name}"] = C.haversine_km(slat, slon, obs["lat"].values, obs["lon"].values)
    obs.to_csv(os.path.join(a.out, "iquam_obs_in_box.csv"), index=False)

    # (a) sub-region x phase x day/night x platform group
    def summarise(g):
        return pd.Series(dict(n_obs=len(g),
                              n_platform_days=g[["platform_id", "date"]].drop_duplicates().shape[0],
                              n_platforms=g["platform_id"].nunique(),
                              n_months_with_obs=g[["year", "month"]].drop_duplicates().shape[0]))
    t1 = (obs[obs["daynight"] != "twilight"]
          .groupby(["subregion", "phase", "daynight", "platform_group"]).apply(summarise).reset_index())
    t1["meets_Li_n50"] = t1["n_obs"] >= C.LI_MIN_ACCURACY
    t1.to_csv(os.path.join(a.out, "table1_region_phase_daynight.csv"), index=False)

    # (b) station x radius x phase, drifters vs all platforms
    rows = []
    for name, *_ in C.STATIONS:
        d = obs[f"dist_km::{name}"]
        for r in RADII_KM:
            sub = obs[d <= r]
            for ph in C.PHASE_ORDER:
                for dn in ("day", "night", "all"):
                    s = sub[sub["phase"] == ph]
                    if dn != "all":
                        s = s[s["daynight"] == dn]
                    dr = s[s["platform_group"] == "drifter"]
                    rows.append(dict(station=name, radius_km=r, phase=ph, daynight=dn,
                                     n_all_platforms=len(s), n_drifter=len(dr),
                                     n_drifter_ql5=int((dr["quality_level"] >= 5).sum()),
                                     n_drifter_platform_days=dr[["platform_id", "date"]].drop_duplicates().shape[0],
                                     meets_rule_n50=bool(len(dr) >= C.LI_MIN_ACCURACY)))
    t2 = pd.DataFrame(rows)
    t2.to_csv(os.path.join(a.out, "table2_station_radius_phase.csv"), index=False)

    # compact console summary
    print("\n=== Drifter observations by sub-region x phase x day/night (upper bound on matchups) ===")
    piv = (t1[t1["platform_group"] == "drifter"]
           .pivot_table(index=["subregion", "phase"], columns="daynight", values="n_obs", fill_value=0))
    print(piv.reindex(C.PHASE_ORDER, level="phase").to_string())
    print("\n=== Drifter observations within 5 / 25 km of each station, all phases pooled ===")
    pooled = t2[t2["daynight"] == "all"].groupby(["station", "radius_km"])["n_drifter"].sum().unstack()
    print(pooled[[5, 25]].to_string())
    rule = t2[(t2["radius_km"] == 25) & (t2["daynight"] != "all")]
    print(f"\nStation x phase x day/night units with >= 50 drifter observations within 25 km: "
          f"{int(rule['meets_rule_n50'].sum())} of {len(rule)}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        dr = t1[t1["platform_group"] == "drifter"].copy()
        dr["col"] = dr["phase"] + "\n" + dr["daynight"]
        hm = dr.pivot_table(index="subregion", columns="col", values="n_obs", fill_value=0)
        order = [f"{p}\n{dn}" for p in C.PHASE_ORDER for dn in ("day", "night")]
        hm = hm.reindex(columns=[c for c in order if c in hm.columns])
        fig, ax = plt.subplots(figsize=(10, 2 + 0.6 * len(hm)))
        im = ax.imshow(hm.values, cmap="viridis", aspect="auto")
        ax.set_xticks(range(hm.shape[1]), hm.columns, fontsize=8)
        ax.set_yticks(range(hm.shape[0]), hm.index)
        for i in range(hm.shape[0]):
            for j in range(hm.shape[1]):
                v = int(hm.values[i, j])
                ax.text(j, i, v, ha="center", va="center",
                        color="red" if v < C.LI_MIN_ACCURACY else "white", fontsize=8)
        ax.set_title(f"iQuam drifter observations (QL>={a.ql_min}); red = below n=50")
        fig.colorbar(im, ax=ax, label="observations")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "fig_drifter_counts.png"), dpi=200)
    except Exception as e:  # plotting is optional
        print("Figure skipped:", e)

    print(f"\nOutputs written to {a.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", help="print the structure of one iQuam file and exit")
    ap.add_argument("--iquam-dir", help="folder with iQuam monthly .nc files")
    ap.add_argument("--gebco", default=None, help="GEBCO_2023 netCDF subset covering the box")
    ap.add_argument("--ql-min", type=int, default=4)
    ap.add_argument("--out", default="out/step1a")
    a = ap.parse_args()

    if a.inspect:
        inspect(a.inspect)
        return
    if not a.iquam_dir:
        sys.exit("Give --iquam-dir or --inspect")

    files = sorted(glob.glob(os.path.join(a.iquam_dir, "*.nc")))
    if not files:
        sys.exit(f"No .nc files in {a.iquam_dir}")
    os.makedirs(a.out, exist_ok=True)

    parts, sources = [], set()
    for f in files:
        df, src = read_one(f, C.SCOPING_BOX, a.ql_min)
        print(f"{os.path.basename(f)}: {len(df)} records in box (QL>={a.ql_min})")
        if len(df):
            parts.append(df)
            sources.add(src)
    if not parts:
        sys.exit("No records in the box. Check SCOPING_BOX and the files.")
    obs = pd.concat(parts, ignore_index=True)
    print("Platform codes decoded from:", ", ".join(sorted(sources)))

    summarise_and_save(obs, a.gebco, a.out, a.ql_min)


if __name__ == "__main__":
    main()
