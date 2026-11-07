"""Step 6b: checks on primary test 1 that analysis plan v1.0 does not include (EXPLORATORY).

Run after stage 1 of Step 6 on 11 Oct 2026, to understand why the Freedman-Lane p-value and the
wild cluster bootstrap interval disagree. None of these numbers replaces a pre-registered result.

1. Independent re-fit of beta with statsmodels OLS (dummies) and CR1 cluster-robust SE by year.
2. Freedman-Lane with WHOLE-YEAR permutation: one permutation of the ten years applied to every
   calendar month and station, so dependence between months of the same year is kept.
3. Leave-one-year-out beta.
4. Correlation of reduced-model residuals between months of the same year (z and u).
"""
import argparse, os
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import step6_link_analysis as S6

ap = argparse.ArgumentParser()
ap.add_argument("--hindcast", required=True); ap.add_argument("--uncertainty", required=True)
ap.add_argument("--n-perm", type=int, default=9999); ap.add_argument("--seed", type=int, default=20261012)
ap.add_argument("--out", required=True)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(a.seed)
hind, unc = pd.read_csv(a.hindcast), pd.read_csv(a.uncertainty)
rows = []
for method in S6.METHODS:
    d = S6.build_station_months(hind, unc, method)
    d["cell"] = d["station"] + "|" + d["month"].astype(str); d["t"] = d["year"] - 2015
    f = smf.ols("z ~ C(cell) + t + unc_mean_k", data=d).fit(cov_type="cluster", cov_kwds={"groups": d["year"]})
    b, se = f.params["unc_mean_k"], f.bse["unc_mean_k"]
    # whole-year Freedman-Lane
    X = S6.design(d); Q = S6.projector(X); n, k = len(d), Q.shape[1] + 1
    ur = S6.residualise(Q, d["unc_mean_k"].values); zr = S6.residualise(Q, d["z"].values)
    b0, r0 = S6.beta_t(ur, zr); sxx = ur @ ur; t0 = b0 / np.sqrt(r0 @ r0 / (n - k) / sxx)
    years = np.sort(d["year"].unique()); yc = np.searchsorted(years, d["year"].values)
    stc, _ = pd.factorize(d["station"]); mc = d["month"].values - 1
    pos = -np.ones((stc.max() + 1, 12, len(years)), int); pos[stc, mc, yc] = np.arange(n)
    cnt = 0
    for _ in range(a.n_perm):
        p = rng.permutation(len(years)); idx = pos[stc, mc, p[yc]]; idx = np.where(idx < 0, np.arange(n), idx)
        bb, rr = S6.beta_t(ur, S6.residualise(Q, zr[idx]))
        cnt += abs(bb / np.sqrt(rr @ rr / (n - k) / sxx)) >= abs(t0)
    # leave one year out
    loyo = {}
    for y in years:
        dd = d[d["year"] != y]; XX = S6.design(dd); QQ = S6.projector(XX)
        loyo[int(y)] = S6.beta_t(S6.residualise(QQ, dd["unc_mean_k"].values), S6.residualise(QQ, dd["z"].values))[0] * 0.1
    # within-year correlation of residuals between different months
    def iyc(v):
        m = pd.DataFrame({"y": d["year"], "s": d["station"], "mo": d["month"], "v": v})
        w = m.pivot_table(index=["s", "y"], columns="mo", values="v")
        c = w.corr().values; return float(np.nanmean(c[np.triu_indices(12, 1)]))
    rows.append(dict(method=method, beta_per_0p1K=b * 0.1, se_cr1_per_0p1K=se * 0.1, p_cr1_normal=f.pvalues["unc_mean_k"],
                     t_classical=t0, p_FL_whole_year=(cnt + 1) / (a.n_perm + 1),
                     loyo_min=min(loyo.values()), loyo_max=max(loyo.values()),
                     loyo_year_of_min=min(loyo, key=loyo.get), loyo_year_of_max=max(loyo, key=loyo.get),
                     resid_corr_months_same_year_z=iyc(zr), resid_corr_months_same_year_u=iyc(ur)))
    if method == "sarima":
        pd.Series(loyo, name="beta_per_0p1K_without_year").to_csv(os.path.join(a.out, "loyo_sarima.csv"))
out = pd.DataFrame(rows); out.to_csv(os.path.join(a.out, "exploratory_test1_checks.csv"), index=False)
pd.set_option("display.width", 200); print(out.round(4).to_string(index=False))
