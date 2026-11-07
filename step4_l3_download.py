"""Step 4: download Level-3 SST for the northern Bay of Bengal, 2018-2020.

Products (analysis plan v1.0, Section 2):
  metopb : OSI SAF MetOp-B AVHRR L3C, 0.05 deg   (NASA PO.DAAC, needs Earthdata login)
  aqua   : NOAA ACSPO MODIS Aqua L3C, 0.02 deg   (NOAA CoastWatch THREDDS, no login)
  terra  : NOAA ACSPO MODIS Terra L3C, 0.02 deg  (NOAA CoastWatch THREDDS, no login)

For every file, only the box 18-23 N, 86-93 E is read from the server (OPeNDAP for the
ACSPO files, HTTP range reads for PO.DAAC), and the variables needed later are kept:
sea_surface_temperature, quality_level, sst_dtime, l2p_flags, sses_bias,
sses_standard_deviation. Each subset is written as a small compressed netCDF file.
The run can be stopped and restarted; finished files are skipped.

  python step4_l3_download.py --product aqua --inspect               # show the catalogue tree
  python step4_l3_download.py --product metopb --out OUT --workers 4
  python step4_l3_download.py --inventory --out OUT                  # coverage summary to send back
  python step4_l3_download.py --product aqua --local-dir FILES --out OUT   # offline test
"""

import argparse
import glob
import os
import re
import shutil
import sys
import tempfile
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import xarray as xr

import config as C

BOX = dict(lat_min=18.0, lat_max=23.0, lon_min=86.0, lon_max=93.0)
STATION_BOX = dict(lat_min=20.0, lat_max=22.5, lon_min=89.0, lon_max=92.5)
KEEP_VARS = ["sea_surface_temperature", "quality_level", "sst_dtime", "l2p_flags",
             "sses_bias", "sses_standard_deviation"]
ACSPO_CATALOG = "https://coastwatch.noaa.gov/thredds/socd/coastwatch/acspo/catalog_sst_acspo_modis_ran.xml"
PRODUCTS = {
    "metopb": dict(source="earthaccess", short_name="AVHRR_SST_METOP_B_GLB-OSISAF-L3C-v1.0",
                   own=[], other=[]),
    "aqua": dict(source="thredds", catalog=ACSPO_CATALOG,
                 own=["modis_a", "aqua"], other=["modis_t", "terra"]),
    "terra": dict(source="thredds", catalog=ACSPO_CATALOG,
                  own=["modis_t", "terra"], other=["modis_a", "aqua"]),
}


# ------------------------------------------------------------------ helpers
def file_date(name):
    """GHRSST file names start with YYYYMMDDhhmmss; fall back to the first YYYYMMDD found."""
    m = re.search(r"(?<!\d)((?:19|20)\d{6})(\d{6})?(?!\d)", os.path.basename(name))
    if not m:
        return None
    try:
        return pd.Timestamp(m.group(1))
    except ValueError:
        return None


def keep_dataset(name, spec, start, end):
    low = name.lower()
    if not low.endswith((".nc", ".nc4")) or "l3c" not in low:
        return False
    if spec["own"] and not any(t in low for t in spec["own"]):
        return False
    d = file_date(name)
    return d is not None and start <= d <= end


def keep_ref(name, ctx, spec, start, end):
    """Decide whether to descend into a catalogue folder; returns (keep, new_context)."""
    clean = name.strip("/").strip()
    low = clean.lower()
    ctx = dict(ctx)
    if any(t in low for t in spec["other"]) and not any(t in low for t in spec["own"]):
        return False, ctx
    if "l2p" in low or "l3u" in low or "l3s" in low:
        return False, ctx
    if re.fullmatch(r"(19|20)\d\d", clean):
        year = int(clean)
        ctx["year"] = year
        return start.year <= year <= end.year, ctx
    if re.fullmatch(r"\d{3}", clean) and "year" in ctx:
        try:
            d = pd.Timestamp(f"{ctx['year']}-01-01") + pd.Timedelta(days=int(clean) - 1)
        except ValueError:
            return True, ctx
        return start <= d <= end, ctx
    if re.fullmatch(r"\d{2}", clean) and "year" in ctx:
        m = int(clean)
        if 1 <= m <= 12:
            first = pd.Timestamp(f"{ctx['year']}-{m:02d}-01")
            last = first + pd.offsets.MonthEnd(0)
            return not (last < start or first > end), ctx
    return True, ctx


def siphon_lister(url):
    from siphon.catalog import TDSCatalog
    try:
        cat = TDSCatalog(url)
    except Exception:
        if not url.endswith(".xml"):
            raise
        cat = TDSCatalog(url[:-4] + ".html")
    datasets = {}
    for name, ds in cat.datasets.items():
        acc = {k.lower(): v for k, v in ds.access_urls.items()}
        datasets[name] = acc.get("opendap")
    refs = {name: ref.href for name, ref in cat.catalog_refs.items()}
    return datasets, refs


def crawl(root, spec, start, end, lister=siphon_lister, max_depth=7, log=print):
    found, stack, n_cat = [], [(root, 0, {})], 0
    while stack:
        url, depth, ctx = stack.pop()
        try:
            datasets, refs = lister(url)
        except Exception as e:
            log(f"  catalogue failed, skipped: {url} ({e})")
            continue
        n_cat += 1
        if n_cat % 50 == 0:
            log(f"  {n_cat} catalogues read, {len(found)} files found")
        for name, opendap in datasets.items():
            if opendap and keep_dataset(name, spec, start, end):
                found.append((name, opendap))
        if depth < max_depth:
            for name, href in refs.items():
                keep, new_ctx = keep_ref(name, ctx, spec, start, end)
                if keep:
                    stack.append((href, depth + 1, new_ctx))
    return found


def inspect_tree(root, levels=3):
    stack = [(root, 0)]
    while stack:
        url, depth = stack.pop(0)
        datasets, refs = siphon_lister(url)
        pad = "  " * depth
        print(f"{pad}{url}")
        for n in list(datasets)[:5]:
            print(f"{pad}  file: {n}")
        names = list(refs)
        print(f"{pad}  folders ({len(names)}): {names[:12]}{' ...' if len(names) > 12 else ''}")
        if depth + 1 < levels:
            for n in names[:3]:
                stack.append((refs[n], depth + 1))


def earthaccess_granules(spec, start, end):
    import earthaccess
    earthaccess.login(strategy="netrc")
    res = earthaccess.search_data(short_name=spec["short_name"],
                                  temporal=(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d 23:59:59")),
                                  bounding_box=(BOX["lon_min"], BOX["lat_min"], BOX["lon_max"], BOX["lat_max"]))
    out = []
    for g in res:
        links = [l for l in g.data_links(access="external") if l.endswith((".nc", ".nc4"))]
        if links:
            out.append((os.path.basename(links[0]), links[0]))
    return out


def granule_list(product, start, end, out, local_dir=None):
    spec = PRODUCTS[product]
    if local_dir:
        files = sorted(glob.glob(os.path.join(local_dir, "*.nc")))
        return [(os.path.basename(f), f) for f in files
                if keep_dataset(os.path.basename(f), spec, start, end) or spec["source"] == "earthaccess"]
    cache = os.path.join(out, f"granules_{product}.csv")
    if os.path.exists(cache):
        g = pd.read_csv(cache)
        print(f"Using cached list {cache}: {len(g)} files")
        return list(zip(g["name"], g["url"]))
    print(f"Listing {product} files on the server ...")
    if spec["source"] == "earthaccess":
        found = earthaccess_granules(spec, start, end)
    else:
        found = crawl(spec["catalog"], spec, start, end)
    found = sorted(set(found))
    pd.DataFrame(found, columns=["name", "url"]).to_csv(cache, index=False)
    print(f"{len(found)} files listed and saved to {cache}")
    return found


# ------------------------------------------------------------------ subset and write
def coord(ds, options):
    for o in options:
        if o in ds.coords or o in ds.variables:
            return o
    raise KeyError(f"none of {options} in file")


def index_range(values, lo, hi):
    idx = np.where((values >= lo) & (values <= hi))[0]
    if idx.size == 0:
        return None
    return slice(int(idx.min()), int(idx.max()) + 1)


def subset(ds):
    latn, lonn = coord(ds, ["lat", "latitude"]), coord(ds, ["lon", "longitude"])
    lat = ds[latn].values
    lon = ds[lonn].values
    lon = np.where(lon > 180, lon - 360, lon)
    ys = index_range(lat, BOX["lat_min"], BOX["lat_max"])
    xs = index_range(lon, BOX["lon_min"], BOX["lon_max"])
    if ys is None or xs is None:
        raise ValueError("box not inside this file's grid")
    keep = [v for v in KEEP_VARS if v in ds.variables]
    if "sea_surface_temperature" not in keep:
        raise KeyError(f"sea_surface_temperature missing; variables: {list(ds.data_vars)}")
    sub = ds[keep].isel({ds[latn].dims[0]: ys, ds[lonn].dims[0]: xs})
    sub = sub.rename({latn: "lat", lonn: "lon"}) if (latn, lonn) != ("lat", "lon") else sub
    sub.attrs = {k: v for k, v in ds.attrs.items()
                 if k in ("title", "id", "product_version", "processing_level", "platform", "sensor",
                          "institution", "time_coverage_start", "time_coverage_end", "day_or_night",
                          "naming_authority", "references", "doi", "DOI")}
    return sub.load()


def write(sub, path):
    enc = {}
    for v in sub.data_vars:
        e = {"zlib": True, "complevel": 4}
        if v == "sea_surface_temperature":
            e.update(dtype="int16", scale_factor=0.01, add_offset=273.15, _FillValue=np.int16(-32768))
        elif v in ("sses_bias", "sses_standard_deviation"):
            e.update(dtype="int16", scale_factor=0.001, add_offset=0.0, _FillValue=np.int16(-32768))
        elif v == "quality_level":
            e.update(dtype="int8", _FillValue=np.int8(-128))
        elif v == "l2p_flags":
            e.update(dtype="int32", _FillValue=np.int32(-2147483648))
        enc[v] = e
    for v in sub.data_vars:  # drop stale packing attributes copied from the source
        for k in ("scale_factor", "add_offset", "_FillValue", "missing_value", "valid_min", "valid_max"):
            sub[v].attrs.pop(k, None)
    tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False).name
    sub.to_netcdf(tmp, encoding=enc)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    shutil.move(tmp, path)


_FS = None


def _init_worker(source):
    global _FS
    if source == "earthaccess":
        import earthaccess
        earthaccess.login(strategy="netrc")
        _FS = earthaccess.get_fsspec_https_session()


def open_remote(url, source):
    """Return (dataset, handle_to_close). PO.DAAC files are read by HTTP range requests;
    if that fails the whole file is fetched to a temporary path."""
    if source == "earthaccess" and _FS is not None:
        fh = _FS.open(url)
        try:
            return xr.open_dataset(fh, engine="h5netcdf", decode_timedelta=False), fh
        except Exception:
            fh.close()
            tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False).name
            _FS.get(url, tmp)
            return xr.open_dataset(tmp, decode_timedelta=False), tmp
    return xr.open_dataset(url, decode_timedelta=False), None


def fetch_one(task):
    name, url, out_path, source = task
    last = None
    for attempt in range(1, 4):
        handle = None
        try:
            ds, handle = open_remote(url, source)
            with ds:
                write(subset(ds), out_path)
            return name, "ok", ""
        except Exception as e:
            last = e
            time.sleep(5 * attempt)
        finally:
            if isinstance(handle, str):
                os.remove(handle)
            elif handle is not None:
                handle.close()
    return name, "failed", str(last)[:300]


def out_path_for(out, product, name):
    d = file_date(name)
    year = d.year if d is not None else "unknown"
    stem = os.path.splitext(os.path.basename(name))[0]
    return os.path.join(out, product, str(year), stem + "_nbob.nc")


def download(product, start, end, out, workers, local_dir=None, months=None):
    spec = PRODUCTS[product]
    files = granule_list(product, start, end, out, local_dir)
    if months:
        # accuracy fallback (plan v1.0, Section 3): only the months of the phase that is short of matchups
        files = [(n, u) for n, u in files if file_date(n) is not None and file_date(n).month in set(months)]
    tasks = [(n, u, out_path_for(out, product, n), "local" if local_dir else spec["source"]) for n, u in files]
    todo = [t for t in tasks if not os.path.exists(t[2])]
    print(f"{product}: {len(tasks)} files in the window, {len(tasks) - len(todo)} already done, {len(todo)} to fetch")
    fail_path = os.path.join(out, f"failed_{product}.csv")
    failed, t0 = [], time.time()
    if todo:
        with Pool(workers, initializer=_init_worker, initargs=("local" if local_dir else spec["source"],)) as pool:
            for k, (name, status, msg) in enumerate(pool.imap_unordered(fetch_one, todo), 1):
                if status != "ok":
                    failed.append(dict(name=name, error=msg))
                    print(f"  FAILED {name}: {msg}")
                if k % 50 == 0 or k == len(todo):
                    rate = (time.time() - t0) / k
                    print(f"  {k}/{len(todo)} done, {rate:.1f} s per file, about {rate * (len(todo) - k) / 60:.0f} min left")
    pd.DataFrame(failed, columns=["name", "error"]).to_csv(fail_path, index=False)
    print(f"{product}: finished, {len(failed)} failed (listed in {fail_path}; run the same command again to retry)")


# ------------------------------------------------------------------ inventory
def inventory(out):
    rows, attrs = [], {}
    for product in PRODUCTS:
        files = sorted(glob.glob(os.path.join(out, product, "*", "*_nbob.nc")))
        for f in files:
            try:
                with xr.open_dataset(f, decode_timedelta=False) as ds:
                    if product not in attrs:
                        attrs[product] = dict(ds.attrs, example_file=os.path.basename(f),
                                              variables=", ".join(ds.data_vars))
                    sst = ds["sea_surface_temperature"].squeeze()
                    ql = ds["quality_level"].squeeze() if "quality_level" in ds else None
                    st = sst.where((ds.lat >= STATION_BOX["lat_min"]) & (ds.lat <= STATION_BOX["lat_max"]) &
                                   (ds.lon >= STATION_BOX["lon_min"]) & (ds.lon <= STATION_BOX["lon_max"]), drop=True)
                    good = st.notnull()
                    if ql is not None:
                        q = ql.sel(lat=st.lat, lon=st.lon)
                        good = good & (q >= 4)
                    rows.append(dict(product=product, file=os.path.basename(f), date=file_date(f),
                                     n_pixels_station_box=int(st.size),
                                     n_valid_ql4_station_box=int(good.sum()),
                                     has_sst_dtime="sst_dtime" in ds, has_quality_level=ql is not None,
                                     day_or_night=str(ds.attrs.get("day_or_night", ""))))
            except Exception as e:
                rows.append(dict(product=product, file=os.path.basename(f), date=file_date(f), error=str(e)[:200]))
    if not rows:
        sys.exit(f"No subset files found under {out}")
    inv = pd.DataFrame(rows)
    inv.to_csv(os.path.join(out, "inventory_files.csv"), index=False)
    inv["month"] = pd.to_datetime(inv["date"]).dt.to_period("M").astype(str)
    inv["frac_valid"] = inv["n_valid_ql4_station_box"] / inv["n_pixels_station_box"]
    mon = inv.groupby(["product", "month"]).agg(n_files=("file", "size"),
                                                mean_frac_valid_station_box=("frac_valid", "mean")).reset_index()
    mon.to_csv(os.path.join(out, "inventory_monthly.csv"), index=False)
    with open(os.path.join(out, "product_attributes.txt"), "w") as fh:
        for p, a in attrs.items():
            fh.write(f"== {p}\n" + "\n".join(f"{k}: {v}" for k, v in a.items()) + "\n\n")
    print("\n=== Files per product and year ===")
    inv["year"] = pd.to_datetime(inv["date"]).dt.year
    print(inv.pivot_table(index="product", columns="year", values="file", aggfunc="count", fill_value=0).to_string())
    inv["phase"] = pd.to_datetime(inv["date"]).dt.month.map(C.PHASE_OF_MONTH)
    print("\n=== Mean share of station-box pixels (land included) that are valid with QL >= 4, by phase ===")
    print(inv.pivot_table(index="product", columns="phase", values="frac_valid", aggfunc="mean")
          .reindex(columns=C.PHASE_ORDER).round(3).to_string())
    print("\nWith sst_dtime:", inv.groupby("product")["has_sst_dtime"].mean().round(2).to_dict())
    if "error" in inv:
        print("Unreadable files:", int(inv["error"].notna().sum()))
    print(f"\nInventory written to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", choices=list(PRODUCTS))
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2020-12-31")
    ap.add_argument("--out", default="out/step4_l3")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--months", type=int, nargs="+", default=None,
                    help="keep only files from these calendar months (e.g. 6 7 8 9 for the monsoon)")
    ap.add_argument("--inspect", action="store_true", help="print the ACSPO catalogue tree and exit")
    ap.add_argument("--inventory", action="store_true", help="summarise the downloaded subsets")
    ap.add_argument("--local-dir", default=None, help="offline test with files already on disk")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    vfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VERSION")
    print("Package version:", open(vfile).read().strip() if os.path.exists(vfile) else "unknown")
    if a.inventory:
        inventory(a.out)
        return
    if not a.product:
        sys.exit("Give --product metopb, aqua or terra")
    if a.inspect:
        inspect_tree(PRODUCTS[a.product].get("catalog", ACSPO_CATALOG))
        return
    download(a.product, pd.Timestamp(a.start), pd.Timestamp(a.end), a.out, a.workers, a.local_dir, a.months)


if __name__ == "__main__":
    main()
