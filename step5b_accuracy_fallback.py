"""Step 5b: accuracy fallback of analysis plan v1.0, Section 3 (fixed before any L3 data were processed).

"If a regional unit has fewer than 50 matchups after cloud screening, its Accuracy window is extended back
to 2016-2017 (iQuam scoping repeated for those years); if it is still short, that unit's Accuracy is
descriptive only."

Inputs
  --main  Step 5 output folder for 2018-2020 (with check_ql5/ and check_r10/ inside)
  --ext   Step 5 output folder for the extension run on 2016-2017 (same code, same bathymetry file;
          only the months of the failing phase need to be downloaded); check_ql5/ inside is used for
          the quality-level-5 check when present
  --out   folder for the final RQ1 tables

What it does
  1. Finds the offshore units (product x phase x day/night) with fewer than 50 matchups in --main.
  2. Adds the 2016-2017 matchups of those units only, and recomputes A1-A3, the drifter-pass
     bootstrap intervals (same seed) and the leave-one-drifter-out ranges for them. Other units keep
     their 2018-2020 values unchanged. A unit that still has fewer than 50 matchups stays descriptive.
  3. C2 for a product x phase uses the extended window on both sides when either side was extended.
  4. Re-scores and re-ranks the station units with the same code as Step 5 (rank_units), for the main
     run and for the two checks (quality level 5 with its own extension; 10 km radius, whose matchups
     are those of the main run).
  5. Writes a complete Step 5 folder to --out (availability, C1 and daily coverage copied from --main),
     so Step 6 can read it with --step5 OUT.

Availability, C1 and every forecast quantity are untouched: the window extension concerns accuracy only.

  python step5b_accuracy_fallback.py --main OUT5 --ext OUT5_EXT --out OUT5_FINAL
"""

import argparse
import os
import shutil

import pandas as pd

import config as C
import step5_reliability as S

KEYS = ["product", "zone", "phase", "daynight"]
EXT_YEARS = (2016, 2017)
COPY = ["availability_station.csv", "availability_station_grid005.csv", "c1_station.csv",
        "daily_coverage_station.csv", "monthly_rmsd_station.csv", "run_info.txt", "log_main.txt"]


def read_matchups(folder):
    path = os.path.join(folder, "matchups.csv")
    if not os.path.exists(path) or os.path.getsize(path) < 10:
        return pd.DataFrame()
    m = pd.read_csv(path, parse_dates=["time"])
    return m


def fallback(main_dir, ext_dir, out, log):
    os.makedirs(out, exist_ok=True)
    mt = read_matchups(main_dir)
    ext = read_matchups(ext_dir) if ext_dir else pd.DataFrame()
    if len(ext):
        ext = ext[ext["time"].dt.year.between(*EXT_YEARS)]
    acc_main = pd.read_csv(os.path.join(main_dir, "accuracy_regional.csv"))
    ci_main = pd.read_csv(os.path.join(main_dir, "accuracy_regional_ci.csv"))
    short = acc_main[(acc_main["zone"] == "offshore") & (acc_main["n"] < C.LI_MIN_ACCURACY)][KEYS]
    log(f"  offshore units below {C.LI_MIN_ACCURACY} matchups in 2018-2020: {len(short)}")

    ext_units = ext.merge(short, on=KEYS) if len(ext) and len(short) else pd.DataFrame(columns=mt.columns)
    final = pd.concat([mt, ext_units], ignore_index=True) if len(ext_units) else mt.copy()

    # C2: a product x phase with any extended side uses the extended window on both sides
    ext_pp = short[["product", "phase"]].drop_duplicates()
    ext_c2 = ext[ext["zone"] == "offshore"].merge(ext_pp, on=["product", "phase"]) if len(ext) else pd.DataFrame()
    c2_input = pd.concat([mt, ext_c2], ignore_index=True) if len(ext_c2) else mt

    acc, _ = S.accuracy_tables(final)
    _, c2 = S.accuracy_tables(c2_input)
    n_add = ext_units.groupby(KEYS).size().rename("n_added_2016_2017").reset_index() if len(ext_units) \
        else pd.DataFrame(columns=KEYS + ["n_added_2016_2017"])
    acc = acc.merge(n_add, on=KEYS, how="left")
    acc["n_added_2016_2017"] = acc["n_added_2016_2017"].fillna(0).astype(int)
    # window label: a unit is "extended" only when 2016-2017 matchups were actually added to it
    acc["window"] = "2018-2020"
    acc = acc.merge(short.assign(_short=True), on=KEYS, how="left")
    is_short = acc["_short"].fillna(False).astype(bool)
    acc.loc[is_short & (acc["n_added_2016_2017"] > 0), "window"] = "2016-2017 + 2018-2020"
    acc.loc[is_short & (acc["n_added_2016_2017"] == 0), "window"] = "2018-2020 (2016-2017 searched, none found)"
    acc = acc.drop(columns="_short")
    c2["window"] = "2018-2020"
    if len(ext_c2):
        c2_pp = ext_c2[["product", "phase"]].drop_duplicates().assign(zone="offshore", _ext=True)
        c2 = c2.merge(c2_pp, on=["product", "zone", "phase"], how="left")
        c2.loc[c2["_ext"].fillna(False).astype(bool), "window"] = "2016-2017 + 2018-2020"
        c2 = c2.drop(columns="_ext")

    # intervals: recomputed for every unit from the final matchups with the Step 5 seed. For units without added
    # matchups this reproduces the Step 5 bootstrap intervals exactly; it also corrects the drifter counts and the
    # leave-one-drifter-out ranges where Step 5 (before 2026-10-13d) read one drifter's id once as a number and once
    # as text, and so counted it twice (see read_iquam in step5_reliability.py).
    ci = S.accuracy_ci(final)
    chk = ci_main.merge(ci, on=KEYS, suffixes=("_step5", ""))
    for col in ("n_drifters", "bias_ci_low", "bias_ci_high", "rmse_ci_low", "rmse_ci_high"):
        diff = chk[(chk["zone"] == "offshore") & ((chk[col + "_step5"] - chk[col]).abs() > 1e-9)]
        if len(diff):
            log(f"  {col} differs from Step 5 in {len(diff)} offshore units: "
                + "; ".join(f"{r['product']} {r['phase']} {r['daynight']} {r[col + '_step5']:.3g} -> {r[col]:.3g}"
                            for _, r in diff.iterrows()))
    acc_pass, c2_pass = S.accuracy_tables(S.one_per_pass(final))

    final.to_csv(os.path.join(out, "matchups.csv"), index=False)
    acc.to_csv(os.path.join(out, "accuracy_regional.csv"), index=False)
    c2.to_csv(os.path.join(out, "c2_regional.csv"), index=False)
    ci.to_csv(os.path.join(out, "accuracy_regional_ci.csv"), index=False)
    acc_pass.to_csv(os.path.join(out, "accuracy_regional_one_per_pass.csv"), index=False)
    c2_pass.to_csv(os.path.join(out, "c2_regional_one_per_pass.csv"), index=False)
    for f in COPY:
        src = os.path.join(main_dir, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out, f))

    avail = pd.read_csv(os.path.join(main_dir, "availability_station.csv"))
    c1 = pd.read_csv(os.path.join(main_dir, "c1_station.csv"))
    a005 = pd.read_csv(os.path.join(main_dir, "availability_station_grid005.csv"))
    scored, status, rec, sh, sens = S.rank_units(avail, c1, acc, a005, out)

    added = int(len(ext_units))
    with open(os.path.join(out, "fallback_info.txt"), "w") as fh:
        fh.write(f"main {main_dir}\next {ext_dir}\nunits below {C.LI_MIN_ACCURACY} in 2018-2020: {len(short)}\n"
                 f"2016-2017 matchups in the extension run: {len(ext)}\n"
                 f"2016-2017 matchups added to them: {added}\n"
                 f"units still below {C.LI_MIN_ACCURACY} (descriptive only): "
                 f"{int(((acc['zone'] == 'offshore') & (acc['n'] < C.LI_MIN_ACCURACY)).sum())}\n")
    show = acc[(acc["zone"] == "offshore")].merge(short, on=KEYS)
    log(f"  2016-2017 matchups in the extension run: {len(ext)}; added to short units: {added}")
    if len(show):
        log(show[["product", "phase", "daynight", "window", "n", "n_added_2016_2017", "n_drifters", "n_drifter_passes",
                  "bias", "A2", "A3", "A_valid"]].round(3).to_string(index=False))
    log("  screening:\n" + status.to_string())
    if len(sh):
        log("  shares (%):\n" + sh.pivot_table(index="product", columns="scenario", values="share_pct").round(1).to_string())
    return acc, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--ext", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    log = print
    log("== main run (quality level >= 4, 25 km)")
    acc, _ = fallback(a.main, a.ext, a.out, log)

    q5_main = os.path.join(a.main, "check_ql5")
    if os.path.isdir(q5_main):
        q5_ext = os.path.join(a.ext, "check_ql5")
        log("== check: quality level 5")
        fallback(q5_main, q5_ext if os.path.isdir(q5_ext) else None, os.path.join(a.out, "check_ql5"), log)

    r10 = os.path.join(a.main, "check_r10")
    if os.path.isdir(r10):
        # the radius changes availability and C1 only; accuracy is that of the final main run
        log("== check: 10 km radius")
        out = os.path.join(a.out, "check_r10")
        os.makedirs(out, exist_ok=True)
        for f in COPY:
            src = os.path.join(r10, f)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(out, f))
        avail = pd.read_csv(os.path.join(r10, "availability_station.csv"))
        c1 = pd.read_csv(os.path.join(r10, "c1_station.csv"))
        a005 = pd.read_csv(os.path.join(r10, "availability_station_grid005.csv"))
        acc.to_csv(os.path.join(out, "accuracy_regional.csv"), index=False)
        _, status, _, sh, _ = S.rank_units(avail, c1, acc, a005, out)
        log("  screening:\n" + status.to_string())
        if len(sh):
            log("  shares (%):\n" + sh.pivot_table(index="product", columns="scenario", values="share_pct").round(1).to_string())
    log(f"Final RQ1 tables written to {a.out}")


if __name__ == "__main__":
    main()
