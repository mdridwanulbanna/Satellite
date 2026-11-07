"""Exploratory check of Test 1 (not in analysis plan v1.0; run 12 Oct 2026).

Question: is the negative Test 1 estimate explained by the L4 analysis relaxing toward its background
when few observations are available, which would damp the station anomaly in high-uncertainty months?
Same structure as Test 1: station x calendar-month fixed effects and a linear year trend, 840 station-months
(2015-2024) and 2100 (2000-2024); inference by whole-year permutation (4,999, seed 20261012).
Outcomes: |anomaly| (detrended within station x month), |anomaly|/SD, and |month-to-month change of anomaly|.

  python step6c_exploratory_anomaly.py --out results/step6/exploratory
"""
import argparse, os
import numpy as np
import pandas as pd


def fe_perm(df, y, n_perm=4999, seed=20261012):
    X = pd.get_dummies(df.station + "_" + df.month.astype(str)).astype(float)
    X["t"] = df.year - df.year.min()
    Xv = X.values
    res = lambda v: v - Xv @ np.linalg.lstsq(Xv, v, rcond=None)[0]
    ru, ry = res(df.unc_mean_k.values), res(df[y].values)
    beta = (ru @ ry) / (ru @ ru)
    years = np.sort(df.year.unique())
    yi = np.searchsorted(years, df.year.values)
    key = {(s, m, yy): i for i, (s, m, yy) in enumerate(zip(df.station, df.month, df.year))}
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(years))
        idx = np.array([key[(s, m, years[perm[k]])] for s, m, k in zip(df.station, df.month, yi)])
        cnt += abs((ru @ ry[idx]) / (ru @ ru)) >= abs(beta)
    return dict(beta_per_0p1K=beta * 0.1, p_whole_year=(cnt + 1) / (n_perm + 1), partial_r=float(np.corrcoef(ru, ry)[0, 1]), n=len(df))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sst", default="results/step1b/station_monthly_sst.csv")
    ap.add_argument("--uncertainty", default="results/step1c/station_monthly_l4_uncertainty.csv")
    ap.add_argument("--out", default="results/step6/exploratory")
    a = ap.parse_args()
    sst = pd.read_csv(a.sst, parse_dates=["date"]).set_index("date").loc["2000":"2024"]
    u = pd.read_csv(a.uncertainty, parse_dates=["date"])
    d = sst.reset_index().melt(id_vars="date", var_name="station", value_name="sst").merge(
        u[["station", "date", "unc_mean_k"]], on=["station", "date"]).sort_values(["station", "date"])
    d["month"], d["year"] = d.date.dt.month, d.date.dt.year
    d["a"] = d.groupby(["station", "month"], group_keys=False).apply(
        lambda g: g.sst - np.polyval(np.polyfit(g.year, g.sst, 1), g.year))
    d["abs_anom"] = d.a.abs()
    d["abs_anom_std"] = d.abs_anom / d.groupby(["station", "month"]).a.transform("std")
    d["abs_anom_change"] = d.groupby("station").a.diff().abs()
    rows = []
    for period, df in [("2015-2024", d[d.year >= 2015]), ("2000-2024", d[d.year >= 2001])]:
        df = df.dropna(subset=["abs_anom_change"]).reset_index(drop=True)
        for y in ["abs_anom", "abs_anom_std", "abs_anom_change"]:
            rows.append(dict(period=period, outcome=y, **fe_perm(df, y)))
    out = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    out.to_csv(os.path.join(a.out, "anomaly_check.csv"), index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
