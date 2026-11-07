"""Step 5: RQ1 reliability metrics, screening and product scoring (analysis plan v1.0, Section 3).

Inputs (all produced by earlier steps):
  --l3     folder with the Step 4 subsets: <l3>/{metopb,aqua,terra}/<year>/*_nbob.nc
  --iquam  folder with the Step 2 iQuam box files box_YYYYMM.csv (quality level >= 4 records)
  --l4     daily C3S L4 SST for the station area (downloaded here if not given)
  --bathy  bathymetry netCDF (GEBCO_2023 fetched from a THREDDS server if not given,
           ETOPO1 from NOAA ERDDAP as fallback)

Outputs in --out:
  matchups.csv                   every satellite-drifter matchup (Li et al. rules)
  accuracy_regional.csv          A1-A3 by product x zone x phase x day/night
  c2_regional.csv                C2 by product x zone x phase
  accuracy_regional_one_per_pass.csv, c2_regional_one_per_pass.csv
                                 same, one matchup per drifter per satellite file (check)
  accuracy_regional_ci.csv       95% intervals for bias, RMSE and r from a bootstrap over drifter
                                 passes (drifter x satellite file), plus leave-one-drifter-out ranges
                                 and the share of the largest drifter (amendment 1 of the plan)
  availability_station_grid005.csv
                                 B1-B3 with every product on a common 0.05 deg grid (a 0.05 deg cell
                                 counts as observed when at least half of its pixels are valid);
                                 resolution check, amendment 1
  daily_coverage_station.csv     daily valid coverage, product x station x date x day/night
  availability_station.csv       B1-B3 by product x station x phase x day/night
  monthly_rmsd_station.csv       monthly L3-L4 RMSD at each station
  c1_station.csv                 C1 by product x station x phase x day/night
  scoring_units.csv              all units: raw metrics, validity, scores, scenario scores
  recommendations.csv            winners per comparison unit and scenario
  recommendation_shares.csv      tie-adjusted shares per scenario
  sensitivity.csv                winner-set agreement under weight perturbations
  screening_status.csv           unit counts by screening outcome
  run_info.txt                   settings, data sources, product file counts

No forecast error is read here: this step answers RQ1 only.

  python step5_reliability.py --l3 L3DIR --iquam IQDIR --out OUT5
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import xarray as xr

import config as C
from step1b_l4_station_series import coord_name
from step4_l3_download import BOX, file_date

PRODUCTS = ["metopb", "aqua", "terra"]
STATION_AREA_BOX = dict(lat_min=20.1, lat_max=22.6, lon_min=89.1, lon_max=92.7)
MATCH_HOURS = 3.0
MATCH_KM = 5.0
EFFECTIVE_COVERAGE = 0.10
W_A = dict(A1=0.25, A2=0.55, A3=0.20)
W_B = dict(B1=0.50, B2=0.30, B3=0.20)
W_C = dict(C1=1.0)
SCENARIOS = {  # Li et al., Table III
    "scientific": (0.60, 0.15, 0.25),
    "operational": (0.20, 0.55, 0.25),
    "long_term": (0.20, 0.25, 0.55),
    "balanced": (0.33, 0.33, 0.34),
}
SCENARIO_PERTURBATIONS = {  # Li et al., Table IV
    "scientific": [(0.60, 0.20, 0.20), (0.55, 0.15, 0.30)],
    "operational": [(0.20, 0.60, 0.20), (0.25, 0.55, 0.20)],
    "long_term": [(0.30, 0.20, 0.50), (0.30, 0.15, 0.55)],
    "balanced": [(0.40, 0.30, 0.30), (0.30, 0.40, 0.30)],
}
WITHIN_PERTURBATIONS = {
    "equal_within": (dict(A1=1 / 3, A2=1 / 3, A3=1 / 3), dict(B1=1 / 3, B2=1 / 3, B3=1 / 3)),
    "rmse_emphasis": (dict(A1=0.15, A2=0.70, A3=0.15), W_B),
}
DIRECTION = dict(A1=-1, A2=-1, A3=1, B1=1, B2=-1, B3=-1, C1=-1)
# Sensitivity only: smallest difference treated as meaningful. Min-max scaling over two or three
# products turns any difference, however small, into a 0-100 gap; with these floors a smaller
# spread gives scores close to each other instead.
DIFFERENCE_FLOOR = dict(A1=0.05, A2=0.05, A3=0.02, B1=0.05, B2=3.0, B3=1.0, C1=0.05)
TIE_TOLERANCE = 1.0  # points on the 0-100 scale
COMMON_GRID_DEG = 0.05      # amendment 1: availability on a common grid (MetOp-B resolution)
COMMON_GRID_MIN_FRAC = 0.5  # share of valid native pixels needed for a common-grid cell to count
N_BOOT_ACC = 999            # amendment 1: bootstrap replicates over drifter passes


# ------------------------------------------------------------------ inputs
def read_l3(path, ql_min):
    with xr.open_dataset(path, decode_timedelta=False) as ds:
        sst = ds["sea_surface_temperature"].squeeze("time", drop=False).values.astype(float)
        if sst.ndim == 3:
            sst = sst[0]
        if np.nanmean(sst) > 200:
            sst = sst - 273.15
        ql = ds["quality_level"].values.squeeze() if "quality_level" in ds else np.full(sst.shape, 5)
        t0 = pd.Timestamp(ds["time"].values.ravel()[0])
        if "sst_dtime" in ds:
            dt = ds["sst_dtime"].values.squeeze()
            if np.issubdtype(dt.dtype, np.timedelta64):
                dt = dt.astype("timedelta64[s]").astype(float)
            dt = dt.astype(float)
        else:
            dt = np.zeros(sst.shape)
        sses = ds["sses_bias"].values.squeeze().astype(float) if "sses_bias" in ds else np.full(sst.shape, np.nan)
        lat = ds["lat"].values.astype(float)
        lon = ds["lon"].values.astype(float)
    valid = np.isfinite(sst) & np.isfinite(dt) & (np.nan_to_num(ql, nan=-1) >= ql_min)
    secs = np.where(valid, dt, 0).astype("int64")
    ptime = np.datetime64(t0.to_datetime64(), "s") + secs.astype("timedelta64[s]")
    return dict(sst=sst, ql=ql, valid=valid, time=ptime, sses=sses, lat=lat, lon=lon)


def list_l3(l3_dir, product, start, end):
    files = sorted(glob.glob(os.path.join(l3_dir, product, "*", "*_nbob.nc")))
    out = []
    for f in files:
        d = file_date(f)
        if d is not None and start - pd.Timedelta(days=1) <= d <= end + pd.Timedelta(days=1):
            out.append(f)
    return out


def read_iquam(iq_dir, start, end, ql_min):
    files = sorted(glob.glob(os.path.join(iq_dir, "box_*.csv")))
    if not files:
        sys.exit(f"No box_*.csv files in {iq_dir}")
    df = pd.concat([pd.read_csv(f, parse_dates=["time"]) for f in files if os.path.getsize(f) > 10],
                   ignore_index=True)
    df = df[(df["platform_group"] == "drifter") & (df["quality_level"] >= ql_min)]
    # monthly files can read the same id as a number in one month and as text in another; use one form
    df["platform_id"] = df["platform_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    df = df[(df["time"] >= start) & (df["time"] < end + pd.Timedelta(days=1))]
    df = df[(df["lat"].between(BOX["lat_min"], BOX["lat_max"])) & (df["lon"].between(BOX["lon_min"], BOX["lon_max"]))]
    return df.reset_index(drop=True)


def get_l4(path, out, start, end):
    if path:
        ds = xr.open_dataset(path)
    else:
        cache = os.path.join(out, "l4_daily_stationarea.nc")
        if not os.path.exists(cache):
            import copernicusmarine as cm
            b = STATION_AREA_BOX
            ds = cm.open_dataset(dataset_id=C.L4_DATASET_ID, variables=["analysed_sst", "mask"],
                                 minimum_latitude=b["lat_min"], maximum_latitude=b["lat_max"],
                                 minimum_longitude=b["lon_min"], maximum_longitude=b["lon_max"],
                                 start_datetime=start.strftime("%Y-%m-%d"), end_datetime=end.strftime("%Y-%m-%d"))
            ds.load().to_netcdf(cache)
        ds = xr.open_dataset(cache)
    latn, lonn = coord_name(ds, ["latitude", "lat"]), coord_name(ds, ["longitude", "lon"])
    ds = ds.sortby([latn, lonn]).sel(time=slice(start, end + pd.Timedelta(hours=23)))
    sst = ds["analysed_sst"].values.astype(float)
    if np.nanmean(sst) > 200:
        sst = sst - 273.15
    water = np.isfinite(sst).any(axis=0)
    if "mask" in ds:
        m = ds["mask"].values
        water &= (m == 1).all(axis=0) if m.ndim == 3 else (m == 1)
    days = pd.DatetimeIndex(ds["time"].values).normalize()
    return dict(sst=sst, lat=ds[latn].values.astype(float), lon=ds[lonn].values.astype(float),
                water=water, day_index={d: i for i, d in enumerate(days)})


def get_bathy(path, out):
    """Elevation in metres (negative below sea level) on a regular grid."""
    if path:
        src, ds = path, xr.open_dataset(path)
    else:
        cache = os.path.join(out, "bathymetry_nbob.nc")
        if not os.path.exists(cache):
            ds = None
            try:
                from siphon.catalog import TDSCatalog
                cat = TDSCatalog("https://ihthredds.ihcantabria.com/thredds/catalog/Bathymetry/Global/gebco_2023/catalog.xml")
                name = next(n for n in cat.datasets if "gebco" in n.lower() and n.endswith(".nc"))
                acc = {k.lower(): v for k, v in cat.datasets[name].access_urls.items()}
                remote = xr.open_dataset(acc["opendap"])
                latn, lonn = coord_name(remote, ["lat", "latitude"]), coord_name(remote, ["lon", "longitude"])
                ds = remote.sel({latn: slice(17.5, 23.5), lonn: slice(85.5, 93.5)}).load()
                ds.attrs["bathymetry_source"] = f"GEBCO_2023 ({name}) via IHCantabria THREDDS"
            except Exception as e:
                print("GEBCO_2023 not reachable, using ETOPO1 from NOAA ERDDAP:", e)
                import requests
                url = ("https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.nc?"
                       "altitude[(17.5):1:(23.5)][(85.5):1:(93.5)]")
                tmp = cache + ".tmp"
                with open(tmp, "wb") as fh:
                    fh.write(requests.get(url, timeout=300).content)
                ds = xr.open_dataset(tmp).load()
                ds.attrs["bathymetry_source"] = "ETOPO1 via NOAA ERDDAP (etopo180)"
                os.remove(tmp)
            ds.to_netcdf(cache)
        src, ds = cache, xr.open_dataset(cache)
    var = next(v for v in ["elevation", "altitude", "z", "Band1"] if v in ds)
    latn, lonn = coord_name(ds, ["lat", "latitude"]), coord_name(ds, ["lon", "longitude"])
    ds = ds.sortby([latn, lonn])
    return dict(z=ds[var].values.astype(float), lat=ds[latn].values.astype(float),
                lon=ds[lonn].values.astype(float), source=ds.attrs.get("bathymetry_source", src))


def nearest_idx(grid, values):
    """Index of the nearest grid value (grid sorted ascending or descending)."""
    grid = np.asarray(grid)
    asc = grid[0] <= grid[-1]
    g = grid if asc else grid[::-1]
    i = np.clip(np.searchsorted(g, values), 1, len(g) - 1)
    i = np.where(np.abs(values - g[i - 1]) <= np.abs(values - g[i]), i - 1, i)
    return i if asc else len(g) - 1 - i


def depth_at(bathy, lat, lon):
    z = bathy["z"][nearest_idx(bathy["lat"], np.asarray(lat)), nearest_idx(bathy["lon"], np.asarray(lon))]
    return -z  # depth positive downwards


# ------------------------------------------------------------------ station areas
def station_areas(lat, lon, l4, bathy, radius_km):
    """For a product grid: per station, flat indices of water pixels within radius_km."""
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    li = nearest_idx(l4["lat"], LAT.ravel())
    lj = nearest_idx(l4["lon"], LON.ravel())
    inside_l4 = ((LAT.ravel() >= l4["lat"].min() - 0.03) & (LAT.ravel() <= l4["lat"].max() + 0.03) &
                 (LON.ravel() >= l4["lon"].min() - 0.03) & (LON.ravel() <= l4["lon"].max() + 0.03))
    water = l4["water"][li, lj] & inside_l4 & (depth_at(bathy, LAT.ravel(), LON.ravel()) > 0)
    areas = {}
    for name, _, _, glat, glon in C.STATIONS:
        d = C.haversine_km(glat, glon, LAT.ravel(), LON.ravel())
        areas[name] = np.where((d <= radius_km) & water)[0]
    return areas, li, lj


def runs_below(cov, dates, threshold):
    """Longest run and number of runs of days below threshold, within contiguous date segments."""
    longest, events, run = 0, 0, 0
    prev = None
    for d, c in zip(dates, cov):
        if prev is not None and (d - prev).days != 1:
            run = 0
        if c < threshold:
            if run == 0:
                events += 1
            run += 1
            longest = max(longest, run)
        else:
            run = 0
        prev = d
    return longest, events


# ------------------------------------------------------------------ main passes
def process_product(product, files, l4, bathy, buoys, args, log):
    """One pass over a product's files: coverage, L3-L4 differences and matchup candidates."""
    cov_hits = {}            # (station, date, dn) -> bool array over area pixels
    rmsd_acc = {}            # (station, yyyymm, dn) -> [sum_sq, n]
    cand = []                # matchup candidates
    grid_key, areas = None, None
    btime = buoys["time"].values.astype("datetime64[s]")
    blat, blon = buoys["lat"].values, buoys["lon"].values
    t0 = time.time()
    for k, f in enumerate(files, 1):
        try:
            g = read_l3(f, args.ql_min)
        except Exception as e:
            log(f"  unreadable, skipped: {os.path.basename(f)} ({e})")
            continue
        key = (len(g["lat"]), len(g["lon"]), round(float(g["lat"][0]), 4), round(float(g["lon"][0]), 4))
        if key != grid_key:
            areas, li, lj = station_areas(g["lat"], g["lon"], l4, bathy, args.radius_km)
            grid_key = key
            LATf, LONf = [a.ravel() for a in np.meshgrid(g["lat"], g["lon"], indexing="ij")]
            dlat = abs(float(np.median(np.diff(g["lat"]))))
            dlon = abs(float(np.median(np.diff(g["lon"]))))
            win = int(np.ceil(MATCH_KM / (111.0 * min(dlat, dlon)))) + 1
        valid = g["valid"].ravel()
        if not valid.any():
            continue
        ptime = g["time"].ravel()
        sst = g["sst"].ravel()
        sza = np.full(sst.shape, np.nan)
        sza[valid] = C.solar_zenith_deg(ptime[valid], LATf[valid], LONf[valid])
        dn = np.where(sza < C.SZA_DAY_MAX, "day", np.where(sza > C.SZA_NIGHT_MIN, "night", "twilight"))
        pdate = ptime.astype("datetime64[D]")

        # station coverage and L3-L4 differences
        for st, idx in areas.items():
            if idx.size == 0:
                continue
            v = valid[idx]
            if not v.any():
                continue
            for lab in ("day", "night"):
                sel = v & (dn[idx] == lab)
                if not sel.any():
                    continue
                for d in np.unique(pdate[idx][sel]):
                    on_day = sel & (pdate[idx] == d)
                    keyc = (st, pd.Timestamp(d), lab)
                    arr = cov_hits.setdefault(keyc, np.zeros(idx.size, bool))
                    arr |= on_day
                    di = l4["day_index"].get(pd.Timestamp(d))
                    if di is None:
                        continue
                    pix = idx[on_day]
                    ref = l4["sst"][di, li[pix], lj[pix]]
                    diff = sst[pix] - ref
                    diff = diff[np.isfinite(diff)]
                    if diff.size:
                        ym = pd.Timestamp(d).strftime("%Y-%m")
                        acc = rmsd_acc.setdefault((st, ym, lab), [0.0, 0])
                        acc[0] += float(np.sum(diff ** 2))
                        acc[1] += int(diff.size)

        # matchup candidates
        vt = ptime[valid]
        tmin = vt.min() - np.timedelta64(int(MATCH_HOURS * 3600), "s")
        tmax = vt.max() + np.timedelta64(int(MATCH_HOURS * 3600), "s")
        bsel = np.where((btime >= tmin) & (btime <= tmax))[0]
        if bsel.size:
            ii = nearest_idx(g["lat"], blat[bsel])
            jj = nearest_idx(g["lon"], blon[bsel])
            ny, nx = len(g["lat"]), len(g["lon"])
            for b, i0, j0 in zip(bsel, ii, jj):
                ys = np.arange(max(i0 - win, 0), min(i0 + win + 1, ny))
                xs = np.arange(max(j0 - win, 0), min(j0 + win + 1, nx))
                flat = (ys[:, None] * nx + xs[None, :]).ravel()
                flat = flat[valid[flat]]
                if flat.size == 0:
                    continue
                dist = C.haversine_km(blat[b], blon[b], LATf[flat], LONf[flat])
                dt_s = np.abs((ptime[flat] - btime[b]).astype("timedelta64[s]").astype(float))
                ok = (dist <= MATCH_KM) & (dt_s <= MATCH_HOURS * 3600)
                if not ok.any():
                    continue
                fl, dd, tt = flat[ok], dist[ok], dt_s[ok]
                best = np.lexsort((dd, tt))[0]
                p = fl[best]
                cand.append((b, os.path.basename(f), float(tt[best]) / 60.0, float(dd[best]), float(sst[p]),
                             float(g["sses"].ravel()[p]), float(LATf[p]), float(LONf[p]), dn[p],
                             int(np.nan_to_num(g["ql"].ravel()[p], nan=-1)), int(p)))
        if k % 200 == 0 or k == len(files):
            log(f"  {product}: {k}/{len(files)} files, {time.time() - t0:.0f} s")
    return cov_hits, rmsd_acc, cand


def common_grid_cells(areas, lat, lon, deg=COMMON_GRID_DEG):
    """Per station: (inverse index of each area pixel into its common-grid cell, pixels per cell)."""
    nx = len(lon)
    out = {}
    for name, idx in areas.items():
        la, lo = np.asarray(lat)[idx // nx], np.asarray(lon)[idx % nx]
        cid = np.floor((la - 18.0) / deg).astype(int) * 100000 + np.floor((lo - 86.0) / deg).astype(int)
        _, inv, cnt = np.unique(cid, return_inverse=True, return_counts=True)
        out[name] = (inv, cnt)
    return out


def availability_tables(product, cov_hits, areas_size, start, end, cells=None):
    """B1-B3 per station x phase x day/night. With cells (from common_grid_cells), coverage is the
    share of common-grid cells in which at least COMMON_GRID_MIN_FRAC of the native pixels are valid."""
    days = pd.date_range(start, end, freq="D")
    rows_daily, rows_unit = [], []

    def coverage(name, d, lab, n_area):
        hits = cov_hits.get((name, d, lab))
        if hits is None or not n_area:
            return 0.0
        if cells is None:
            return hits.sum() / n_area
        inv, cnt = cells[name]
        frac = np.bincount(inv, weights=hits.astype(float), minlength=cnt.size) / cnt
        return float((frac >= COMMON_GRID_MIN_FRAC).mean())

    for name in C.STATION_NAMES:
        n_area = areas_size.get(name, 0)
        for lab in ("day", "night"):
            cov = np.array([coverage(name, d, lab, n_area) for d in days])
            for d, c in zip(days, cov):
                rows_daily.append((product, name, d, lab, c))
            phase = days.month.map(C.PHASE_OF_MONTH)
            for ph in C.PHASE_ORDER:
                m = np.asarray(phase == ph)
                ds_, cs = days[m], cov[m]
                b2, events = runs_below(cs, ds_, EFFECTIVE_COVERAGE)
                rows_unit.append(dict(product=product, station=name, phase=ph, daynight=lab,
                                      n_area_pixels=n_area if cells is None else int(cells[name][1].size) if name in cells else 0,
                                      n_days=int(m.sum()),
                                      B1=float(cs.mean()) if m.any() else np.nan, B2=b2,
                                      B3=events / m.sum() * 90 if m.any() else np.nan,
                                      n_effective_days=int((cs >= EFFECTIVE_COVERAGE).sum()),
                                      B_valid=bool(n_area > 0 and (cs >= EFFECTIVE_COVERAGE).any())))
    daily = pd.DataFrame(rows_daily, columns=["product", "station", "date", "daynight", "coverage"])
    return daily, pd.DataFrame(rows_unit)


def c1_tables(product, rmsd_acc):
    rows = [dict(product=product, station=s, month=ym, daynight=lab, n_pairs=n,
                 rmsd=np.sqrt(ss / n) if n else np.nan)
            for (s, ym, lab), (ss, n) in rmsd_acc.items()]
    monthly = pd.DataFrame(rows, columns=["product", "station", "month", "daynight", "n_pairs", "rmsd"])
    if monthly.empty:
        return monthly, pd.DataFrame(columns=["product", "station", "phase", "daynight", "n_valid_months", "C1", "C1_valid"])
    monthly["phase"] = pd.to_datetime(monthly["month"]).dt.month.map(C.PHASE_OF_MONTH)
    out = []
    for name in C.STATION_NAMES:
        for ph in C.PHASE_ORDER:
            for lab in ("day", "night"):
                m = monthly[(monthly.station == name) & (monthly.phase == ph) & (monthly.daynight == lab)
                            & (monthly.n_pairs >= C.LI_MIN_C1_MONTH)]
                nv = len(m)
                out.append(dict(product=product, station=name, phase=ph, daynight=lab, n_valid_months=nv,
                                C1=float(np.std(m["rmsd"], ddof=0)) if nv >= 2 else np.nan, C1_valid=nv >= 2))
    return monthly, pd.DataFrame(out)


def matchup_table(product, cand, buoys, bathy):
    cols = ["b", "file", "dt_min", "dist_km", "sat_sst_c", "sses_bias", "pix_lat", "pix_lon", "daynight", "sat_ql", "pix"]
    c = pd.DataFrame(cand, columns=cols)
    if c.empty:
        return c
    c = c.sort_values(["b", "dt_min", "dist_km"]).drop_duplicates("b", keep="first")  # one best per buoy record
    b = buoys.iloc[c["b"].values].reset_index(drop=True)
    c = c.reset_index(drop=True)
    m = pd.concat([b[["time", "lat", "lon", "sst_c", "quality_level", "platform_id"]].rename(
        columns={"sst_c": "buoy_sst_c", "quality_level": "buoy_ql"}), c.drop(columns="b")], axis=1)
    m.insert(0, "product", product)
    m["diff"] = m["sat_sst_c"] - m["buoy_sst_c"]
    m["month"] = m["time"].dt.month
    m["phase"] = m["month"].map(C.PHASE_OF_MONTH)
    m["depth_m"] = depth_at(bathy, m["lat"].values, m["lon"].values)
    m["zone"] = np.where(m["depth_m"] >= C.SHELF_DEPTH_M, "offshore", np.where(m["depth_m"] > 0, "shelf", "land_or_coast"))
    m["pixel_pass"] = m["file"] + ":" + m["pix"].astype(str)
    return m[m["daynight"] != "twilight"]


def one_per_pass(m):
    """Check variant: keep one matchup per drifter and satellite file (closest in time, then distance),
    because Li et al.'s rule keeps one per drifter record and hourly records can share a pixel."""
    if m.empty:
        return m
    return m.sort_values(["dt_min", "dist_km"]).drop_duplicates(["product", "platform_id", "file"], keep="first")


def accuracy_tables(m):
    rows, c2 = [], []
    if m.empty:
        return pd.DataFrame(), pd.DataFrame()
    for (p, z, ph, lab), g in m.groupby(["product", "zone", "phase", "daynight"]):
        d = g["diff"].values
        r = np.corrcoef(g["sat_sst_c"], g["buoy_sst_c"])[0, 1] if len(g) > 2 and g["buoy_sst_c"].std() > 0 else np.nan
        rows.append(dict(product=p, zone=z, phase=ph, daynight=lab, n=len(g),
                         n_unique_pixels=g["pixel_pass"].nunique(), n_drifters=g["platform_id"].nunique(),
                         n_drifter_passes=int(g[["platform_id", "file"]].drop_duplicates().shape[0]),
                         n_months=g["time"].dt.to_period("M").nunique(), bias=float(d.mean()),
                         A1=float(abs(d.mean())), A2=float(np.sqrt(np.mean(d ** 2))), A3=float(r),
                         bias_sses_corrected=float(np.nanmean(d - g["sses_bias"].values)),
                         A_valid=bool(len(g) >= C.LI_MIN_ACCURACY and z == "offshore")))
    acc = pd.DataFrame(rows)
    acc["flag_few_passes"] = acc["n_drifter_passes"] < 10
    for (p, z, ph), g in m.groupby(["product", "zone", "phase"]):
        dd, nn = g[g.daynight == "day"]["diff"], g[g.daynight == "night"]["diff"]
        ok = len(dd) >= C.LI_MIN_C2_SIDE and len(nn) >= C.LI_MIN_C2_SIDE
        c2.append(dict(product=p, zone=z, phase=ph, n_day=len(dd), n_night=len(nn),
                       C2=float(abs(dd.mean() - nn.mean())) if len(dd) and len(nn) else np.nan, C2_valid=ok))
    return acc, pd.DataFrame(c2)


def accuracy_ci(m, n_boot=N_BOOT_ACC, seed=20261011):
    """Amendment 1: 95% percentile intervals for bias, RMSE and r, resampling drifter passes
    (drifter x satellite file) with replacement, because matchups within a pass are not independent."""
    if m.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows = []
    for (p, z, ph, lab), g in m.groupby(["product", "zone", "phase", "daynight"]):
        key = g["platform_id"].astype(str) + "|" + g["file"]
        x, y = g["sat_sst_c"].values, g["buoy_sst_c"].values
        d = x - y
        agg = pd.DataFrame(dict(k=key.values, n=1, d=d, d2=d ** 2, x=x, y=y, xx=x * x, yy=y * y, xy=x * y)).groupby("k").sum()
        S = agg.values  # columns n, d, d2, x, y, xx, yy, xy
        npass = len(S)
        if npass < 2:
            rows.append(dict(product=p, zone=z, phase=ph, daynight=lab, n_passes=npass))
            continue
        idx = rng.integers(0, npass, (n_boot, npass))
        T = S[idx].sum(axis=1)
        n = T[:, 0]
        bias = T[:, 1] / n
        rmse = np.sqrt(T[:, 2] / n)
        vx = T[:, 5] / n - (T[:, 3] / n) ** 2
        vy = T[:, 6] / n - (T[:, 4] / n) ** 2
        cxy = T[:, 7] / n - (T[:, 3] / n) * (T[:, 4] / n)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = cxy / np.sqrt(vx * vy)
        q = lambda v: np.nanpercentile(v, [2.5, 97.5]) if np.isfinite(v).any() else (np.nan, np.nan)
        (bl, bh), (rl, rh), (cl, ch) = q(bias), q(rmse), q(r)
        # leave one drifter out: how much one buoy (and its calibration) drives the result
        lodo_b, lodo_r = [], []
        for pid in g["platform_id"].unique():
            dd_ = d[g["platform_id"].values != pid]
            if dd_.size:
                lodo_b.append(dd_.mean()); lodo_r.append(np.sqrt(np.mean(dd_ ** 2)))
        share = g["platform_id"].value_counts(normalize=True).iloc[0]
        rows.append(dict(product=p, zone=z, phase=ph, daynight=lab, n_matchups=len(g), n_passes=npass,
                         n_drifters=g["platform_id"].nunique(), largest_drifter_share=float(share), n_boot=n_boot,
                         bias_ci_low=bl, bias_ci_high=bh, rmse_ci_low=rl, rmse_ci_high=rh, r_ci_low=cl, r_ci_high=ch,
                         bias_lodo_min=min(lodo_b) if lodo_b else np.nan, bias_lodo_max=max(lodo_b) if lodo_b else np.nan,
                         rmse_lodo_min=min(lodo_r) if lodo_r else np.nan, rmse_lodo_max=max(lodo_r) if lodo_r else np.nan))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ scoring (Li et al. framework)
def normalise(values, direction, floor=0.0):
    """Li et al. min-max scaling to 0-100 within a comparison unit; with floor > 0 the
    denominator is at least floor, so spreads smaller than floor give near-equal scores."""
    v = np.asarray(values, float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    span = max(hi - lo, floor)
    if span < 1e-12:
        return np.full(v.shape, 100.0)
    if direction > 0:
        return 100.0 - (hi - v) / span * 100.0
    return 100.0 - (v - lo) / span * 100.0


def score_units(units, wa=W_A, wb=W_B, wc=W_C, floors=None):
    out = []
    for key, g in units.groupby(["station", "phase", "daynight"]):
        g = g.copy()
        ok = g[g["unit_valid"]]
        g["rankable"] = False
        if len(ok) >= 2:
            s = {m: normalise(ok[m], DIRECTION[m], (floors or {}).get(m, 0.0)) for m in DIRECTION}
            A = sum(wa[k] * s[k] for k in wa)
            B = sum(wb[k] * s[k] for k in wb)
            Cc = sum(wc[k] * s[k] for k in wc)
            g.loc[ok.index, "A_score"] = A
            g.loc[ok.index, "B_score"] = B
            g.loc[ok.index, "C_score"] = Cc
            for name, (a, b, c) in SCENARIOS.items():
                g.loc[ok.index, f"score_{name}"] = a * A + b * B + c * Cc
            g.loc[ok.index, "rankable"] = True
        out.append(g)
    return pd.concat(out)


def winners(scored, weights=None):
    rows = []
    for key, g in scored[scored["rankable"]].groupby(["station", "phase", "daynight"]):
        for name, (a, b, c) in (weights or SCENARIOS).items():
            sc = a * g["A_score"] + b * g["B_score"] + c * g["C_score"]
            top = sc.max()
            win = sorted(g.loc[sc >= top - TIE_TOLERANCE, "product"])
            rows.append(dict(station=key[0], phase=key[1], daynight=key[2], scenario=name,
                             winners="+".join(win), k=len(win)))
    return pd.DataFrame(rows)


def shares(rec):
    if rec.empty:
        return pd.DataFrame()
    rows = []
    for sc, g in rec.groupby("scenario"):
        credit = {}
        for w, k in zip(g["winners"], g["k"]):
            for p in w.split("+"):
                credit[p] = credit.get(p, 0) + 1 / k
        for p, v in credit.items():
            rows.append(dict(scenario=sc, product=p, share_pct=100 * v / len(g), n_units=len(g)))
    return pd.DataFrame(rows)


def agreement(base, other):
    m = base.merge(other, on=["station", "phase", "daynight", "scenario"], suffixes=("", "_alt"))
    return m.groupby("scenario").apply(lambda g: pd.Series(dict(
        agreement_pct=100 * (g["winners"] == g["winners_alt"]).mean(), n_units=len(g),
        changed=int((g["winners"] != g["winners_alt"]).sum())))).reset_index()


def rank_units(avail, c1, acc, avail005, out):
    """Assemble station units (regional offshore accuracy + station availability + station C1), score and
    rank them, run the ranking checks and write scoring_units, screening_status, recommendations,
    recommendation_shares and sensitivity to out. Shared by Step 5 and the accuracy fallback (step5b)."""
    units = avail.merge(c1[["product", "station", "phase", "daynight", "n_valid_months", "C1", "C1_valid"]],
                        on=["product", "station", "phase", "daynight"], how="left")
    acc_off = acc[acc["zone"] == "offshore"][["product", "phase", "daynight", "n", "A1", "A2", "A3", "A_valid"]] \
        if len(acc) else pd.DataFrame(columns=["product", "phase", "daynight", "n", "A1", "A2", "A3", "A_valid"])
    units = units.merge(acc_off.rename(columns={"n": "n_matchups_regional"}), on=["product", "phase", "daynight"], how="left")
    for col in ["A_valid", "C1_valid", "B_valid"]:
        units[col] = units[col].fillna(False).astype(bool)
    units["unit_valid"] = units["A_valid"] & units["B_valid"] & units["C1_valid"]
    have_all = units[["A1", "A2", "A3", "B1", "B2", "B3", "C1"]].notna().all(axis=1)
    units["status"] = np.where(units["unit_valid"], "eligible",
                               np.where(have_all, "complete_but_failed_threshold", "missing_component"))
    scored = score_units(units)
    scored.loc[scored["unit_valid"] & ~scored["rankable"], "status"] = "eligible_not_rankable"
    scored.to_csv(os.path.join(out, "scoring_units.csv"), index=False)
    status = scored.groupby(["product", "status"]).size().unstack(fill_value=0)
    status.to_csv(os.path.join(out, "screening_status.csv"))

    rec = winners(scored)
    rec.to_csv(os.path.join(out, "recommendations.csv"), index=False)
    sh = shares(rec)
    sh.to_csv(os.path.join(out, "recommendation_shares.csv"), index=False)
    sens = []
    variants = {k: (wa, wb, None) for k, (wa, wb) in WITHIN_PERTURBATIONS.items()}
    variants["difference_floor"] = (W_A, W_B, DIFFERENCE_FLOOR)
    for lab, (wa, wb, fl) in variants.items():
        alt = winners(score_units(units, wa, wb, floors=fl))
        if len(rec) and len(alt):
            s = agreement(rec, alt)
            s.insert(0, "test", lab)
            sens.append(s)
    # amendment 1: rankings with every product's availability on the common 0.05 deg grid
    u005 = units.drop(columns=["B1", "B2", "B3", "B_valid", "n_area_pixels", "n_effective_days", "n_days"]).merge(
        avail005[["product", "station", "phase", "daynight", "B1", "B2", "B3", "B_valid"]],
        on=["product", "station", "phase", "daynight"], how="left")
    u005["B_valid"] = u005["B_valid"].fillna(False).astype(bool)
    u005["unit_valid"] = u005["A_valid"] & u005["B_valid"] & u005["C1_valid"]
    alt = winners(score_units(u005))
    if len(rec) and len(alt):
        s = agreement(rec, alt)
        s.insert(0, "test", "availability_common_grid_0.05deg")
        sens.append(s)
    for sc, alts in SCENARIO_PERTURBATIONS.items():
        for wts in alts:
            alt = winners(scored, {sc: wts})
            if len(rec) and len(alt):
                s = agreement(rec[rec.scenario == sc], alt)
                s.insert(0, "test", f"scenario_weights_{wts}")
                sens.append(s)
    sens = pd.concat(sens, ignore_index=True) if sens else pd.DataFrame()
    sens.to_csv(os.path.join(out, "sensitivity.csv"), index=False)
    return scored, status, rec, sh, sens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l3", required=True)
    ap.add_argument("--iquam", required=True)
    ap.add_argument("--l4", default=None)
    ap.add_argument("--bathy", default=None)
    ap.add_argument("--out", default="out/step5")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2020-12-31")
    ap.add_argument("--ql-min", type=int, default=4)
    ap.add_argument("--radius-km", type=float, default=25.0)
    ap.add_argument("--products", nargs="+", default=PRODUCTS)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    start, end = pd.Timestamp(a.start), pd.Timestamp(a.end)
    vfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VERSION")
    version = open(vfile).read().strip() if os.path.exists(vfile) else "unknown"
    log = print
    log(f"Package version: {version}")

    buoys = read_iquam(a.iquam, start, end, a.ql_min)
    log(f"Drifter records (QL >= {a.ql_min}) in the regional box: {len(buoys)} from {buoys['platform_id'].nunique()} drifters")
    l4 = get_l4(a.l4, a.out, start, end)
    log(f"L4 daily fields: {len(l4['day_index'])} days on a {l4['sst'].shape[1]} x {l4['sst'].shape[2]} grid")
    bathy = get_bathy(a.bathy, a.out)
    log(f"Bathymetry: {bathy['source']}")

    dailies, avails, avails005, monthlies, c1s, matchups, info = [], [], [], [], [], [], []
    for p in a.products:
        files = list_l3(a.l3, p, start, end)
        info.append(f"{p}: {len(files)} subset files")
        log(f"\n== {p}: {len(files)} files")
        if not files:
            continue
        cov_hits, rmsd_acc, cand = process_product(p, files, l4, bathy, buoys, a, log)
        g = read_l3(files[0], a.ql_min)
        areas, _, _ = station_areas(g["lat"], g["lon"], l4, bathy, a.radius_km)
        sizes = {k: int(v.size) for k, v in areas.items()}
        log(f"  water pixels within {a.radius_km:.0f} km: {sizes}")
        daily, avail = availability_tables(p, cov_hits, sizes, start, end)
        cells = common_grid_cells(areas, g["lat"], g["lon"])
        _, avail005 = availability_tables(p, cov_hits, sizes, start, end, cells=cells)
        avails005.append(avail005)
        monthly, c1 = c1_tables(p, rmsd_acc)
        mt = matchup_table(p, cand, buoys, bathy)
        log(f"  matchups: {len(mt)} (offshore {int((mt['zone'] == 'offshore').sum()) if len(mt) else 0})")
        dailies.append(daily); avails.append(avail); monthlies.append(monthly); c1s.append(c1); matchups.append(mt)

    daily = pd.concat(dailies, ignore_index=True)
    avail = pd.concat(avails, ignore_index=True)
    avail005 = pd.concat(avails005, ignore_index=True)
    avail005.to_csv(os.path.join(a.out, "availability_station_grid005.csv"), index=False)
    monthly = pd.concat(monthlies, ignore_index=True)
    c1 = pd.concat(c1s, ignore_index=True)
    mt = pd.concat([m for m in matchups if len(m)], ignore_index=True) if any(len(m) for m in matchups) else pd.DataFrame()
    acc, c2 = accuracy_tables(mt)
    acc_pass, c2_pass = accuracy_tables(one_per_pass(mt))
    accuracy_ci(mt).to_csv(os.path.join(a.out, "accuracy_regional_ci.csv"), index=False)
    acc_pass.to_csv(os.path.join(a.out, "accuracy_regional_one_per_pass.csv"), index=False)
    c2_pass.to_csv(os.path.join(a.out, "c2_regional_one_per_pass.csv"), index=False)
    for name, df in [("daily_coverage_station", daily), ("availability_station", avail),
                     ("monthly_rmsd_station", monthly), ("c1_station", c1), ("matchups", mt),
                     ("accuracy_regional", acc), ("c2_regional", c2)]:
        df.to_csv(os.path.join(a.out, f"{name}.csv"), index=False)

    scored, status, rec, sh, sens = rank_units(avail, c1, acc, avail005, a.out)

    with open(os.path.join(a.out, "run_info.txt"), "w") as fh:
        fh.write(f"version {version}\nwindow {a.start} to {a.end}\nql_min {a.ql_min}\nradius_km {a.radius_km}\n"
                 f"matchup +/-{MATCH_HOURS} h, {MATCH_KM} km\nbathymetry {bathy['source']}\n"
                 f"drifter records {len(buoys)}\n" + "\n".join(info) + "\n")

    # console summary
    log("\n=== Regional accuracy, offshore (n, bias, RMSE) ===")
    if len(acc):
        t = acc[acc.zone == "offshore"].pivot_table(index=["product", "daynight"], columns="phase",
                                                     values="n", fill_value=0).reindex(columns=C.PHASE_ORDER)
        log(t.to_string())
        log(acc[acc.zone == "offshore"][["product", "phase", "daynight", "n", "n_drifter_passes", "bias", "A2", "A3",
                                          "A_valid", "flag_few_passes"]]
            .round(3).to_string(index=False))
    else:
        log("No matchups.")
    log("\n=== Availability B1 (mean daily coverage), station x phase, day ===")
    log(avail[avail.daynight == "day"].pivot_table(index=["product", "station"], columns="phase", values="B1")
        .reindex(columns=C.PHASE_ORDER).round(3).to_string())
    log("\n=== B1 on the common 0.05 deg grid minus native B1 (mean over units) ===")
    dd = avail.merge(avail005, on=["product", "station", "phase", "daynight"], suffixes=("", "_005"))
    log((dd["B1_005"] - dd["B1"]).groupby(dd["product"]).mean().round(3).to_string())
    log("\n=== Screening status (units of product x station x phase x day/night) ===")
    log(status.to_string())
    if len(sh):
        log("\n=== Tie-adjusted recommendation shares (%) ===")
        log(sh.pivot_table(index="product", columns="scenario", values="share_pct", fill_value=0).round(1).to_string())
    log(f"\nOutputs written to {a.out}")


if __name__ == "__main__":
    main()
