"""Step 2 (user step): how many in situ SST observations exist near the stations?

Downloads the NOAA iQuam v2.10 monthly files for the L3 window (default 2018-01 to 2020-12)
one at a time, keeps only quality_level >= 4 records inside the scoping box, deletes each
global file, and then writes the scoping tables of step1a_iquam_scoping.py. Bathymetry for
the shelf/offshore split comes from ETOPO1 on NOAA ERDDAP (GEBCO_2023 replaces it in the
final analysis).

  python iquam_scope_auto.py --out out/step2_iquam
  python iquam_scope_auto.py --local-dir my_iquam_files --out out/test     # offline test

The run can be stopped and restarted; months already processed are skipped.
"""

import argparse
import glob
import os
import re
import time

import pandas as pd

import config as C
from step1a_iquam_scoping import read_one, summarise_and_save

LIST_URL = "https://www.star.nesdis.noaa.gov/pub/sod/sst/iquam/v2.10/"
ETOPO_URL = ("https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.nc?"
             "altitude[(17.5):1:(23.5)][(85.5):1:(93.5)]")


def month_list(start, end):
    return [p.strftime("%Y%m") for p in pd.period_range(start, end, freq="M")]


def remote_names():
    import requests
    html = requests.get(LIST_URL, timeout=120).text
    return sorted(set(re.findall(r'(\d{6}-STAR-L2i_GHRSST-SST-iQuam-V2\.10[^"<>/]*?\.nc)', html)))


def download(url, path, tries=3):
    import requests
    for k in range(1, tries + 1):
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8 << 20):
                        f.write(chunk)
            return
        except Exception as e:
            print(f"  download attempt {k} failed: {e}")
            time.sleep(10 * k)
    raise RuntimeError(f"could not download {url}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01")
    ap.add_argument("--end", default="2020-12")
    ap.add_argument("--ql-min", type=int, default=4)
    ap.add_argument("--local-dir", default=None, help="use files already on disk (no download)")
    ap.add_argument("--out", default="out/step2_iquam")
    a = ap.parse_args()
    box_dir = os.path.join(a.out, "box_monthly")
    os.makedirs(box_dir, exist_ok=True)

    months = month_list(a.start, a.end)
    if a.local_dir:
        files = {os.path.basename(f)[:6]: f for f in glob.glob(os.path.join(a.local_dir, "*.nc"))}
    else:
        names = remote_names()
        files = {n[:6]: LIST_URL + n for n in names}
        print(f"{len(names)} files listed on the iQuam server")

    sources = set()
    for ym in months:
        cache = os.path.join(box_dir, f"box_{ym}.csv")
        if os.path.exists(cache):
            continue
        if ym not in files:
            print(f"{ym}: no file found, skipped")
            continue
        t0 = time.time()
        if a.local_dir:
            path = files[ym]
        else:
            path = os.path.join(a.out, "tmp_iquam.nc")
            download(files[ym], path)
        df, src = read_one(path, C.SCOPING_BOX, a.ql_min)
        if src:
            sources.add(src)
        df.to_csv(cache, index=False)
        if not a.local_dir:
            os.remove(path)
        print(f"{ym}: {len(df):6d} records in box, "
              f"{int((df['platform_group'] == 'drifter').sum()) if len(df) else 0:6d} drifter, "
              f"{time.time() - t0:5.0f} s")
    if sources:
        print("Platform codes decoded from:", ", ".join(sorted(sources)))

    parts = [pd.read_csv(f, parse_dates=["time"]) for f in sorted(glob.glob(os.path.join(box_dir, "box_*.csv")))]
    parts = [p for p in parts if len(p)]
    if not parts:
        raise SystemExit("No in situ records in the box for the requested months.")
    obs = pd.concat(parts, ignore_index=True)

    bathy = os.path.join(a.out, "etopo1_nbob.nc")
    if not os.path.exists(bathy) and not a.local_dir:
        try:
            download(ETOPO_URL, bathy, tries=2)
        except Exception as e:
            print("Bathymetry download failed, shelf/offshore split skipped:", e)
    summarise_and_save(obs, bathy if os.path.exists(bathy) else None, os.path.join(a.out, "tables"), a.ql_min)


if __name__ == "__main__":
    main()
