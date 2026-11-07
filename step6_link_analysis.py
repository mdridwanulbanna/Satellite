"""Step 6: link between observational support and forecast error (RQ2, RQ3).

Implements Section 4 of analysis plan v1.0 with Amendment 1 (approved 13 Oct 2026) and the RQ3
specification of 13 Oct 2026. This script computes the statistics the plan fixes in advance, so it
refuses to run on real data until the plan has been signed, and refuses stage 2 until plan v1.0 has
been archived with a public timestamp (Amendment 1, item 6):

  python step6_link_analysis.py --signed-off-on 2026-10-11 --hindcast H.csv --uncertainty U.csv \
         --step5 STEP5DIR --plan-archive 10.5281/zenodo.XXXXXXX --out out/step6
  python step6_link_analysis.py --synthetic --out /tmp/test6        # self-test on simulated data

Without --step5 the script runs stage 1 only: primary test 1 and secondary family A, which need no
Level-3 data. The full run later (with --step5) repeats them with the same seed and gives identical
numbers, because test 2, family B, family C and the premise check draw no random numbers, and the
RQ3 bootstrap uses its own generator after stage 1 is finished.

Stage 2 (needs the Step 5 output folder; check_r10/ and check_ql5/ inside it are used when present):
  primary test 2, window 2018-2020 for z (Amendment 1, item 7: option B, 13 Oct 2026); 2015-2024 is secondary
  family B: test 2 for the other methods, products, metrics (B2, B3, C1), radius 10 km, QL 5
  premise check: station-month correlation of MetOp-B B1 and u, 2018-2020 (252 station-months)
  RQ3: one factor at a time (family C), the joint model and the confidence table, as specified on
       13 Oct 2026 before any Step 5 output existed (see rq3() below)

Primary test 1 (direct): z = alpha[station, calendar month] + delta * (year - 2015) + beta * u + e
    z = |forecast error| / training-period interannual SD of that station and calendar month
    u = monthly mean L4 analysis uncertainty (K) of the target month
    p-value: Freedman-Lane permutation (residuals of the model without u permuted as whole
             years within each calendar month, all stations together), 9,999 permutations
    95% interval: wild cluster bootstrap-t, clusters = years, Webb six-point weights
Primary test 2 (indirect): Spearman correlation across the 28 station x phase cells between
    MetOp-B availability B1 and mean z, after removing phase means; exact permutation of the
    seven stations' phase blocks (5,040 orderings)
Primary method: SARIMA. Secondary analyses get Benjamini-Hochberg FDR control (q = 0.10)
within each family.
"""

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

import config as C

METHODS = ["sarima", "lstm", "climatology", "persistence"]
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
TEST2_PRIMARY_YEARS = (2018, 2020)     # Amendment 1, item 7: option B (13 Oct 2026): z from the L3 window
TEST2_FULL_YEARS = (2015, 2024)        # secondary (family B): mean z over the full hindcast
RQ3_Z_YEARS = (2015, 2024)             # RQ3 as specified on 13 Oct 2026; 2018-2020 version reported beside it
L3_YEARS = (2018, 2020)
CLIM_YEARS = (2000, 2024)              # RQ3 interannual SD and seasonal amplitude
RQ3_SCENARIO = "operational"           # confidence table: Li et al. operational weights (A 0.20, B 0.55, C 0.25)


# ------------------------------------------------------------------ data
def build_station_months(hindcast, unc, method):
    h = hindcast.copy()
    h["date"] = pd.to_datetime(h["date"])
    h["z"] = (h["obs"] - h[f"pred_{method}"]).abs() / h["sd_train_calmonth"]
    u = unc.copy()
    u["date"] = pd.to_datetime(u["date"])
    u = u.sort_values(["station", "date"])
    u["u_input12"] = u.groupby("station")["unc_mean_k"].transform(lambda s: s.shift(1).rolling(12).mean())
    d = h.merge(u[["station", "date", "unc_mean_k", "unc_rms_k", "u_input12"]], on=["station", "date"], how="left")
    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.month
    d["phase"] = d["month"].map(C.PHASE_OF_MONTH)
    return d.dropna(subset=["z", "unc_mean_k"]).reset_index(drop=True)


# ------------------------------------------------------------------ test 1
def design(d, year_fe=False):
    cell = pd.Categorical(d["station"] + "|" + d["month"].astype(str))
    X = pd.get_dummies(cell, dtype=float).values
    if year_fe:
        yd = pd.get_dummies(d["year"], dtype=float).values[:, 1:]
        X = np.hstack([X, yd])
    else:
        X = np.hstack([X, (d["year"].values - 2015.0)[:, None]])
    return X


def projector(X):
    """Orthonormal basis of the column space of X (rank-safe), for fast residualising."""
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, S > S.max() * 1e-10]


def residualise(Q, v):
    return v - Q @ (Q.T @ v)


def beta_t(u_res, y_res):
    """OLS slope of y on u after both are residualised on the nuisance design, and the residuals."""
    b = float(u_res @ y_res) / float(u_res @ u_res)
    return b, y_res - b * u_res


def cluster_se(u_res, resid, codes, n_groups, n, k):
    s = np.bincount(codes, weights=u_res * resid, minlength=n_groups)
    c = n_groups / (n_groups - 1) * (n - 1) / (n - k)
    return np.sqrt(c * float(s @ s)) / float(u_res @ u_res)


def freedman_lane(d, xcol, n_perm, rng, year_fe=False):
    X = design(d, year_fe)
    Q = projector(X)
    z = d["z"].values.astype(float)
    x = d[xcol].values.astype(float)
    n, k = len(z), Q.shape[1] + 1
    u_res = residualise(Q, x)
    z_red = residualise(Q, z)                      # residuals of the reduced model
    b_obs, r_obs = beta_t(u_res, z_red)
    sxx = float(u_res @ u_res)
    t_obs = b_obs / np.sqrt(float(r_obs @ r_obs) / (n - k) / sxx)

    # index array: position of (station, month, year) in d, -1 if absent
    st_codes, st_levels = pd.factorize(d["station"])
    years = np.sort(d["year"].unique())
    y_codes = np.searchsorted(years, d["year"].values)
    m_codes = d["month"].values - 1
    pos = -np.ones((len(st_levels), 12, len(years)), int)
    pos[st_codes, m_codes, y_codes] = np.arange(n)
    count = 0
    for _ in range(n_perm):
        perm = np.stack([rng.permutation(len(years)) for _ in range(12)])   # month x year
        idx = pos[st_codes, m_codes, perm[m_codes, y_codes]]
        idx = np.where(idx < 0, np.arange(n), idx)
        b, r = beta_t(u_res, residualise(Q, z_red[idx]))
        count += abs(b / np.sqrt(float(r @ r) / (n - k) / sxx)) >= abs(t_obs)
    p = (count + 1) / (n_perm + 1)

    # wild cluster bootstrap-t (unrestricted), clusters = years, Webb weights
    se_cr = cluster_se(u_res, r_obs, y_codes, len(years), n, k)
    fitted = b_obs * u_res
    ts = np.empty(n_perm)
    for i in range(n_perm):
        w = rng.choice(WEBB, len(years))[y_codes]
        b, r = beta_t(u_res, fitted + residualise(Q, r_obs * w))
        ts[i] = (b - b_obs) / cluster_se(u_res, r, y_codes, len(years), n, k)
    q_hi, q_lo = np.quantile(ts, [0.975, 0.025])
    ci = (b_obs - q_hi * se_cr, b_obs - q_lo * se_cr)
    sd_u = float(np.std(u_res, ddof=k - 1))        # SD of u left after the nuisance terms
    return dict(beta=b_obs, beta_per_0p1K=b_obs * 0.1, ci_low_per_0p1K=ci[0] * 0.1, ci_high_per_0p1K=ci[1] * 0.1,
                sd_u_within=sd_u, beta_per_sd_u=b_obs * sd_u,
                t=t_obs, p_perm=p, n=n, n_years=len(years), se_cluster=se_cr)


# ------------------------------------------------------------------ test 2
def cell_table(d, b1):
    zc = d.groupby(["station", "phase"])["z"].mean().rename("z_mean").reset_index()
    t = zc.merge(b1, on=["station", "phase"], how="inner")
    t["z_dm"] = t["z_mean"] - t.groupby("phase")["z_mean"].transform("mean")
    t["x_dm"] = t["x"] - t.groupby("phase")["x"].transform("mean")
    return t


def station_block_test(t):
    stations = sorted(t["station"].unique())
    xm = t.pivot(index="station", columns="phase", values="x_dm").loc[stations]
    zm = t.pivot(index="station", columns="phase", values="z_dm").loc[stations]
    cols = [c for c in C.PHASE_ORDER if c in xm.columns]
    xm, zm = xm[cols].values, zm[cols].values
    mask = np.isfinite(xm) & np.isfinite(zm)
    rho_obs = stats.spearmanr(xm[mask], zm[mask]).correlation
    count, total = 0, 0
    for perm in itertools.permutations(range(len(stations))):
        xp = xm[list(perm)]
        mk = np.isfinite(xp) & np.isfinite(zm)
        rho = stats.spearmanr(xp[mk], zm[mk]).correlation
        count += abs(rho) >= abs(rho_obs) - 1e-12
        total += 1
    pearson = stats.pearsonr(xm[mask], zm[mask])[0]
    return dict(spearman=rho_obs, pearson=pearson, p_exact=count / total, n_cells=int(mask.sum()),
                n_permutations=total)


def b1_cells(avail, product, metric="B1"):
    """Station x phase value of a Step 5 metric: mean of the day and night units (valid values only)."""
    a = avail[avail["product"] == product]
    if product == "mean":
        a = avail
    if metric == "C1" and "C1_valid" in a.columns:
        a = a[a["C1_valid"].astype(bool)]
    return (a.groupby(["station", "phase"])[metric].mean().rename("x").reset_index())


# ------------------------------------------------------------------ premise check
def premise_check(daily, unc, product="metopb"):
    """Station-month MetOp-B coverage (mean of all day and night values) against u, 2018-2020."""
    dd = daily[daily["product"] == product].copy()
    dd["date"] = pd.to_datetime(dd["date"]).dt.to_period("M").dt.to_timestamp()
    b1m = dd.groupby(["station", "date"])["coverage"].mean().rename("B1_month").reset_index()
    u = unc.copy()
    u["date"] = pd.to_datetime(u["date"])
    m = b1m.merge(u[["station", "date", "unc_mean_k"]], on=["station", "date"], how="inner")
    m = m[m["date"].dt.year.between(*L3_YEARS)].dropna()
    m["month"] = m["date"].dt.month
    rows = [dict(check="raw", n=len(m), pearson=stats.pearsonr(m["B1_month"], m["unc_mean_k"])[0],
                 spearman=stats.spearmanr(m["B1_month"], m["unc_mean_k"]).correlation)]
    for lab, keys in [("station_means_removed", ["station"]),
                      ("station_x_calendar_month_means_removed", ["station", "month"])]:
        x = m["B1_month"] - m.groupby(keys)["B1_month"].transform("mean")
        y = m["unc_mean_k"] - m.groupby(keys)["unc_mean_k"].transform("mean")
        rows.append(dict(check=lab, n=len(m), pearson=stats.pearsonr(x, y)[0], spearman=stats.spearmanr(x, y).correlation))
    return pd.DataFrame(rows), m


# ------------------------------------------------------------------ RQ3 (specified 13 Oct 2026)
def rq3_factors(sst_wide, avail):
    """Factors per station x phase cell.
    sigma: SD across 2000-2024 of monthly L4 SST for each calendar month, averaged over the phase's months (degC)
    amp:   maximum minus minimum of the 2000-2024 monthly climatology at the station (degC)
    f_eff: share of days in the phase, 2018-2020, with MetOp-B coverage >= 0.10 within 25 km, mean of day and night
    """
    s = sst_wide.copy()
    s["date"] = pd.to_datetime(s["date"])
    s = s[s["date"].dt.year.between(*CLIM_YEARS)]
    long = s.melt(id_vars="date", var_name="station", value_name="sst")
    long["month"] = long["date"].dt.month
    by_m = long.groupby(["station", "month"])["sst"].agg(["mean", "std"]).reset_index()
    by_m["phase"] = by_m["month"].map(C.PHASE_OF_MONTH)
    sig = by_m.groupby(["station", "phase"])["std"].mean().rename("sigma").reset_index()
    amp = by_m.groupby("station")["mean"].agg(lambda v: v.max() - v.min()).rename("amp").reset_index()
    a = avail[avail["product"] == "metopb"].copy()
    a["f"] = a["n_effective_days"] / a["n_days"]
    f = a.groupby(["station", "phase"])["f"].mean().rename("f_eff").reset_index()
    return sig.merge(amp, on="station").merge(f, on=["station", "phase"], how="left")


def ols_block_inference(cells, ycol, xcols, rng, n_boot=9999):
    """OLS of y on phase intercepts and standardised factors across station x phase cells.
    p: Freedman-Lane, exact over all orderings of whole-station blocks (each station keeps its four phases).
    95% interval: wild cluster bootstrap-t, clusters = stations, Webb weights."""
    t = cells.dropna(subset=[ycol] + xcols).sort_values(["station", "phase"]).reset_index(drop=True)
    stations = sorted(t["station"].unique())
    st = t["station"].map({s: i for i, s in enumerate(stations)}).values
    ph = pd.get_dummies(pd.Categorical(t["phase"], categories=C.PHASE_ORDER), dtype=float).values
    ph = ph[:, ph.sum(0) > 0]
    Z = {c: ((t[c] - t[c].mean()) / t[c].std(ddof=1)).values for c in xcols}
    y = t[ycol].values.astype(float)
    n, G = len(y), len(stations)
    X_full = np.column_stack([ph] + [Z[c] for c in xcols])
    k = np.linalg.matrix_rank(X_full)
    fit = np.linalg.lstsq(X_full, y, rcond=None)[0]
    res_full = y - X_full @ fit
    r2 = 1 - float(res_full @ res_full) / float(((y - y.mean()) ** 2).sum())
    res_ph = y - ph @ np.linalg.lstsq(ph, y, rcond=None)[0]
    r2_phase = 1 - float(res_ph @ res_ph) / float(((y - y.mean()) ** 2).sum())
    # position of each cell's phase-partner in another station: pos[station, phase]
    pcode = t["phase"].map({p: i for i, p in enumerate(C.PHASE_ORDER)}).values
    pos = -np.ones((G, len(C.PHASE_ORDER)), int)
    pos[st, pcode] = np.arange(n)
    out = []
    for c in xcols:
        others = [Z[o] for o in xcols if o != c]
        Q = projector(np.column_stack([ph] + others))
        x_res = residualise(Q, Z[c])
        y_red = residualise(Q, y)
        b_obs, r_obs = beta_t(x_res, y_red)
        sxx = float(x_res @ x_res)
        t_obs = b_obs / np.sqrt(float(r_obs @ r_obs) / (n - k) / sxx)
        count = total = 0
        for perm in itertools.permutations(range(G)):
            idx = pos[np.asarray(perm)[st], pcode]
            if (idx < 0).any():                       # a station lacks a phase: keep it in place
                idx = np.where(idx < 0, np.arange(n), idx)
            b, r = beta_t(x_res, residualise(Q, y_red[idx]))
            count += abs(b / np.sqrt(float(r @ r) / (n - k) / sxx)) >= abs(t_obs) - 1e-12
            total += 1
        se = cluster_se(x_res, r_obs, st, G, n, k)
        ts = np.empty(n_boot)
        for i in range(n_boot):
            w = rng.choice(WEBB, G)[st]
            b, r = beta_t(x_res, b_obs * x_res + residualise(Q, r_obs * w))
            ts[i] = (b - b_obs) / cluster_se(x_res, r, st, G, n, k)
        q_hi, q_lo = np.quantile(ts, [0.975, 0.025])
        vif = float(((Z[c] - Z[c].mean()) ** 2).sum()) / sxx     # vs phase intercepts and the other factors
        out.append(dict(outcome=ycol, factor=c, coef_per_sd=b_obs, ci_low=b_obs - q_hi * se, ci_high=b_obs - q_lo * se,
                        p_perm_exact=count / total, n_permutations=total, vif=vif,
                        sd_factor=float(t[c].std(ddof=1)), n_cells=n, n_stations=G, r2_model=r2, r2_phase_only=r2_phase))
    return pd.DataFrame(out)


def eta2_phase(cells, ycol):
    t = cells.dropna(subset=[ycol])
    tot = float(((t[ycol] - t[ycol].mean()) ** 2).sum())
    between = float(((t.groupby("phase")[ycol].transform("mean") - t[ycol].mean()) ** 2).sum())
    return between / tot if tot > 0 else np.nan


def support_class(f):
    if not np.isfinite(f):
        return "no data"
    return "good" if f >= 0.50 else "fair" if f >= 0.20 else "poor"


def skill_class(msss):
    if not np.isfinite(msss):
        return "no data"
    return "above climatology" if msss >= 0.10 else "near climatology" if msss > -0.10 else "below climatology"


def confidence_table(hind, avail, rec):
    """Descriptive station x phase guidance. Satellite support: RQ1 winners (operational scenario) and the
    best product's share of effective days. Forecast confidence: SARIMA MSSS against climatology, 2015-2024."""
    h = hind.copy()
    h["date"] = pd.to_datetime(h["date"])
    h["phase"] = h["date"].dt.month.map(C.PHASE_OF_MONTH)
    h["e2_s"] = (h["obs"] - h["pred_sarima"]) ** 2
    h["e2_c"] = (h["obs"] - h["pred_climatology"]) ** 2
    sk = h.groupby(["station", "phase"]).agg(mse_s=("e2_s", "mean"), mse_c=("e2_c", "mean"), n_months=("e2_s", "size")).reset_index()
    sk["rmse_sarima"] = np.sqrt(sk["mse_s"])
    sk["msss_sarima"] = 1 - sk["mse_s"] / sk["mse_c"]
    a = avail.copy()
    a["f"] = a["n_effective_days"] / a["n_days"]
    fp = a.groupby(["station", "phase", "product"])["f"].mean().reset_index()
    best = fp.loc[fp.groupby(["station", "phase"])["f"].idxmax()].rename(columns={"product": "most_available", "f": "f_eff_best"})
    out = sk.merge(best, on=["station", "phase"], how="left")
    if rec is not None and len(rec):
        r = rec[rec["scenario"] == RQ3_SCENARIO].pivot_table(index=["station", "phase"], columns="daynight",
                                                              values="winners", aggfunc="first").reset_index()
        r.columns = [c if c in ("station", "phase") else f"winner_{c}" for c in r.columns]
        out = out.merge(r, on=["station", "phase"], how="left")
    out["satellite_support"] = out["f_eff_best"].map(support_class)
    out["forecast_confidence"] = out["msss_sarima"].map(skill_class)
    out["_p"] = out["phase"].map({p: i for i, p in enumerate(C.PHASE_ORDER)})
    out["_s"] = out["station"].map({s: i for i, s in enumerate(C.STATION_NAMES)})
    return out.sort_values(["_p", "_s"]).drop(columns=["_p", "_s"]).reset_index(drop=True)


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


# ------------------------------------------------------------------ synthetic self-test
def synthetic(rng, beta_true=1.0):
    rows, urows = [], []
    years = range(2015, 2025)
    common = {(y, m): rng.normal(0, 0.03) for y in years for m in range(1, 13)}
    for s_i, (name, *_ ) in enumerate(C.STATIONS):
        for y in years:
            for m in range(1, 13):
                u = 0.30 + 0.12 * (m in (6, 7, 8, 9)) + 0.02 * s_i - 0.004 * (y - 2015) + common[(y, m)] + rng.normal(0, 0.02)
                sd = 0.6 if m in (1, 2, 12) else 0.35
                z = abs(rng.normal(0, 1)) * 0.6 + beta_true * (u - 0.3) + 0.1 * (m in (6, 7))
                rows.append(dict(station=name, date=f"{y}-{m:02d}-01", obs=26.0, pred_sarima=26.0 + z * sd,
                                 pred_lstm=26.0 + z * sd * 1.1, pred_climatology=26.0 + z * sd * 1.05,
                                 pred_persistence=26.0 + z * sd * 3, sd_train_calmonth=sd))
                urows.append(dict(station=name, date=f"{y}-{m:02d}-01", unc_mean_k=u, unc_rms_k=u * 1.02))
    for name, *_ in C.STATIONS:      # earlier months for the 12-month input window
        for m in range(1, 13):
            urows.append(dict(station=name, date=f"2014-{m:02d}-01", unc_mean_k=0.33, unc_rms_k=0.34))
    hind = pd.DataFrame(rows)
    unc = pd.DataFrame(urows)
    # Step 5-like tables. Daily coverage falls with u, so the premise check should be negative.
    ud = unc.assign(date=pd.to_datetime(unc["date"])).set_index(["station", "date"])["unc_mean_k"]
    days = pd.date_range("2018-01-01", "2020-12-31", freq="D")
    daily = []
    for name in C.STATION_NAMES:
        for dn in ("day", "night"):
            mstart = days.to_period("M").to_timestamp()
            uu = ud.loc[name].reindex(mstart).values
            cov = np.clip(1.6 - 3.5 * uu + rng.normal(0, 0.15, len(days)), 0, 1)
            for prod, shift in (("metopb", 0.0), ("aqua", -0.05), ("terra", -0.08)):
                daily.append(pd.DataFrame(dict(product=prod, station=name, date=days, daynight=dn,
                                               coverage=np.clip(cov + shift, 0, 1))))
    daily = pd.concat(daily, ignore_index=True)
    daily["phase"] = daily["date"].dt.month.map(C.PHASE_OF_MONTH)
    g = daily.groupby(["product", "station", "phase", "daynight"])["coverage"]
    avail = g.mean().rename("B1").reset_index()
    avail["n_days"] = g.size().values
    avail["n_effective_days"] = g.apply(lambda v: int((v >= 0.10).sum())).values
    avail["B2"] = rng.uniform(3, 30, len(avail))
    avail["B3"] = rng.uniform(1, 8, len(avail))
    avail["B_valid"] = True
    c1 = avail[["product", "station", "phase", "daynight"]].copy()
    c1["C1"] = rng.uniform(0.1, 0.4, len(c1))
    c1["C1_valid"] = True
    c1["n_valid_months"] = 9
    rec = avail[avail["product"] == "metopb"][["station", "phase", "daynight"]].copy()
    rec["scenario"] = RQ3_SCENARIO
    rec["winners"] = "metopb"
    rec["k"] = 1
    months = pd.date_range("2000-01-01", "2024-12-01", freq="MS")
    sst = pd.DataFrame({"date": months.strftime("%Y-%m-%d")})
    for s_i, name in enumerate(C.STATION_NAMES):
        sst[name] = 28 - (1.5 + 0.2 * s_i) * np.cos(2 * np.pi * (months.month - 1) / 12) + rng.normal(0, 0.3 + 0.03 * s_i, len(months))
    s5 = dict(avail=avail, c1=c1, daily=daily[["product", "station", "date", "daynight", "coverage"]], rec=rec,
              avail_r10=avail.assign(B1=avail["B1"] * 0.97), avail_ql5=avail.assign(B1=avail["B1"] * 0.8), sst=sst)
    return hind, unc, s5


def read_step5(folder, sst_path):
    rd = lambda *p: pd.read_csv(os.path.join(folder, *p)) if os.path.exists(os.path.join(folder, *p)) else None
    s5 = dict(avail=rd("availability_station.csv"), c1=rd("c1_station.csv"), daily=rd("daily_coverage_station.csv"),
              rec=rd("recommendations.csv"), avail_r10=rd("check_r10", "availability_station.csv"),
              avail_ql5=rd("check_ql5", "availability_station.csv"), sst=pd.read_csv(sst_path))
    for k in ("avail", "c1", "daily"):
        if s5[k] is None:
            sys.exit(f"Step 5 folder {folder} lacks the {k} table; run Step 5 first.")
    return s5


def with_c1(avail, c1):
    return avail.merge(c1[["product", "station", "phase", "daynight", "C1", "C1_valid"]],
                       on=["product", "station", "phase", "daynight"], how="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signed-off-on", default=None, help="date the supervisor signed analysis plan v1.0")
    ap.add_argument("--plan-archive", default=None, help="DOI or URL of the public, timestamped copy of plan v1.0 "
                                                          "with Amendment 1 (required for stage 2)")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--archive-deviation", default=None,
                    help="text of a recorded deviation, used in place of --plan-archive when stage 2 is run before the "
                         "public archive (plan run log, Deviation 2)")
    ap.add_argument("--hindcast")
    ap.add_argument("--uncertainty")
    ap.add_argument("--step5")
    ap.add_argument("--sst", default="results/step1b/station_monthly_sst.csv",
                    help="monthly L4 SST at the stations, wide format (RQ3 factors)")
    ap.add_argument("--n-perm", type=int, default=9999)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--out", default="out/step6")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)

    if a.synthetic:
        hind, unc, s5 = synthetic(rng)
        label = "SYNTHETIC SELF-TEST (true beta = 1.0 per K, i.e. 0.1 per 0.1 K)"
    else:
        if not a.signed_off_on:
            sys.exit("Refusing to run on real data: give --signed-off-on DATE once the supervisor has signed "
                     "analysis plan v1.0 (or use --synthetic to test the code).")
        if a.step5 and not (a.plan_archive or a.archive_deviation):
            sys.exit("Refusing to run stage 2: Amendment 1, item 6 requires plan v1.0 to be archived with a public "
                     "timestamp first. Give its DOI or URL with --plan-archive, or the text of a deviation recorded "
                     "in the plan's run log with --archive-deviation.")
        hind = pd.read_csv(a.hindcast)
        unc = pd.read_csv(a.uncertainty)
        s5 = read_step5(a.step5, a.sst) if a.step5 else None
        label = f"REAL DATA, analysis plan v1.0 signed on {a.signed_off_on}"
        if s5 is None:
            label += "; STAGE 1 (test 1 and family A; no Step 5 output yet)"
        else:
            arch = (f"plan archived at {a.plan_archive}" if a.plan_archive
                    else f"NOT ARCHIVED BEFORE THIS RUN: {a.archive_deviation}")
            label += f"; Amendment 1 approved 2026-10-13 (item 7: option B); {arch}; STAGE 2"
    print(label)
    avail = with_c1(s5["avail"], s5["c1"]) if s5 is not None else None

    prim, sec = [], []
    # primary test 1
    d = build_station_months(hind, unc, "sarima")
    r = freedman_lane(d, "unc_mean_k", a.n_perm, rng)
    prim.append(dict(test="1_direct_u", method="sarima", **r))
    # primary test 2 (Amendment 1, item 7, option B: mean z over 2018-2020, the L3 window)
    if avail is not None:
        t = cell_table(d[d["year"].between(*TEST2_PRIMARY_YEARS)], b1_cells(avail, "metopb"))
        prim.append(dict(test="2_indirect_B1_metopb", method="sarima", **station_block_test(t)))

    # secondary family A: test 1 variants
    for method in METHODS:
        dm = build_station_months(hind, unc, method)
        for xcol, yfe, lab in [("unc_mean_k", False, "u_mean"), ("unc_rms_k", False, "u_rms"),
                               ("u_input12", False, "u_input12"), ("unc_mean_k", True, "u_mean_yearFE")]:
            if method == "sarima" and lab == "u_mean":
                continue
            dd = dm.dropna(subset=[xcol])
            r = freedman_lane(dd, xcol, max(999, a.n_perm // 5), rng, year_fe=yfe)
            sec.append(dict(family="A_direct", test=f"{lab}", method=method, estimate=r["beta_per_0p1K"],
                            ci_low=r["ci_low_per_0p1K"], ci_high=r["ci_high_per_0p1K"], p=r["p_perm"], n=r["n"]))
    # secondary family B: test 2 variants (draw no random numbers)
    def add_b(test, method, dm, cells):
        res = station_block_test(cell_table(dm, cells))
        sec.append(dict(family="B_indirect", test=test, method=method, estimate=res["spearman"],
                        p=res["p_exact"], n=res["n_cells"]))
    for method in (METHODS if avail is not None else []):
        dm = build_station_months(hind, unc, method)
        dm = dm[dm["year"].between(*TEST2_PRIMARY_YEARS)]
        for prod in ("metopb", "aqua", "terra", "mean"):
            if method == "sarima" and prod == "metopb":
                continue
            if prod != "mean" and prod not in set(avail["product"]):
                continue
            add_b(f"B1_{prod}", method, dm, b1_cells(avail, prod))
        if method == "sarima":
            dfull = build_station_months(hind, unc, method)
            add_b("B1_metopb_errors_2015_2024", method, dfull[dfull["year"].between(*TEST2_FULL_YEARS)],
                  b1_cells(avail, "metopb"))
            for metric in ("B2", "B3", "C1"):
                add_b(f"{metric}_metopb", method, dm, b1_cells(avail, "metopb", metric))
            for lab, key in (("radius_10km", "avail_r10"), ("ql5", "avail_ql5")):
                if s5.get(key) is not None:
                    add_b(f"B1_metopb_{lab}", method, dm, b1_cells(s5[key], "metopb"))
    if avail is not None:
        # premise check (reported, not tested)
        pc, pc_months = premise_check(s5["daily"], unc)
        pc.to_csv(os.path.join(a.out, "premise_check.csv"), index=False)
        pc_months.to_csv(os.path.join(a.out, "premise_check_station_months.csv"), index=False)
        # RQ3: cells, one factor at a time (family C), joint model, confidence table
        rng3 = np.random.default_rng(a.seed + 3)
        dz = d[d["year"].between(*RQ3_Z_YEARS)]
        cells = rq3_factors(s5["sst"], avail)
        cells = cells.merge(dz.groupby(["station", "phase"])["z"].mean().rename("z_sarima").reset_index(), on=["station", "phase"])
        for metric in ("B1", "C1"):
            cells = cells.merge(b1_cells(avail, "metopb", metric).rename(columns={"x": f"{metric}_metopb"}),
                                on=["station", "phase"], how="left")
        cells.to_csv(os.path.join(a.out, "rq3_cells.csv"), index=False)
        pairs = [("z_sarima", "sigma"), ("z_sarima", "amp"), ("z_sarima", "f_eff"),
                 ("B1_metopb", "sigma"), ("B1_metopb", "amp"),          # f_eff is built from the same coverage as B1
                 ("C1_metopb", "sigma"), ("C1_metopb", "amp"), ("C1_metopb", "f_eff")]
        for yc, xc in pairs:
            tt = cells[["station", "phase", yc, xc]].rename(columns={yc: "z", xc: "x"}).dropna()
            tt["z_mean"] = tt["z"]
            tt["z_dm"] = tt["z"] - tt.groupby("phase")["z"].transform("mean")
            tt["x_dm"] = tt["x"] - tt.groupby("phase")["x"].transform("mean")
            res = station_block_test(tt)
            sec.append(dict(family="C_rq3_one_factor", test=f"{yc}~{xc}", method="sarima" if yc == "z_sarima" else "-",
                            estimate=res["spearman"], p=res["p_exact"], n=res["n_cells"]))
        phase_rows = []
        for yc in ("z_sarima", "B1_metopb", "C1_metopb"):
            pm = cells.groupby("phase")[yc].mean().reindex(C.PHASE_ORDER)
            phase_rows.append(dict(outcome=yc, eta2_phase=eta2_phase(cells, yc), **{f"mean_{k}": v for k, v in pm.items()}))
        pd.DataFrame(phase_rows).to_csv(os.path.join(a.out, "rq3_phase.csv"), index=False)
        joint = ols_block_inference(cells, "z_sarima", ["sigma", "amp", "f_eff"], rng3, n_boot=a.n_perm)
        joint.to_csv(os.path.join(a.out, "rq3_joint.csv"), index=False)
        # beside it: the same model with mean z over the L3 window (reported, not part of family C)
        z18 = d[d["year"].between(*L3_YEARS)].groupby(["station", "phase"])["z"].mean().rename("z_sarima_2018_2020")
        cells18 = cells.merge(z18.reset_index(), on=["station", "phase"])
        joint18 = ols_block_inference(cells18, "z_sarima_2018_2020", ["sigma", "amp", "f_eff"], rng3, n_boot=a.n_perm)
        joint18.to_csv(os.path.join(a.out, "rq3_joint_z2018_2020.csv"), index=False)
        conf = confidence_table(hind[pd.to_datetime(hind["date"]).dt.year.between(*RQ3_Z_YEARS)], avail, s5["rec"])
        conf.to_csv(os.path.join(a.out, "confidence_table.csv"), index=False)

    prim = pd.DataFrame(prim)
    sec = pd.DataFrame(sec)
    sec["q_bh"] = sec.groupby("family")["p"].transform(lambda p: pd.Series(bh(p.values), index=p.index))
    prim.to_csv(os.path.join(a.out, "link_primary.csv"), index=False)
    sec.to_csv(os.path.join(a.out, "link_secondary.csv"), index=False)
    with open(os.path.join(a.out, "run_info.txt"), "w") as fh:
        fh.write(f"{label}\nseed {a.seed}\npermutations {a.n_perm}\n")
    pd.set_option("display.width", 160)
    print("\n=== Primary tests ===")
    print(prim.round(4).to_string(index=False))
    print("\n=== Secondary analyses (FDR within family) ===")
    print(sec.round(4).to_string(index=False))
    if avail is not None:
        print("\n=== Premise check: MetOp-B B1 vs u, station-months 2018-2020 ===")
        print(pc.round(3).to_string(index=False))
        print("\n=== RQ3 joint model: mean SARIMA z ~ phase + sigma + amp + f_eff (per SD of factor) ===")
        print(joint.round(3).to_string(index=False))
        print("\n=== Confidence table ===")
        print(conf.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
