"""Exploratory (not in the analysis plan): why do the two prespecified inferences of test 1 disagree?

Test 1 (SARIMA, 840 station-months) gave a Freedman-Lane permutation p of 0.042, while the 95% wild cluster
bootstrap-t interval by year (-0.97 to 0.26 z per 0.1 K) includes zero. This script runs, on the same data:

  1. the score of each year: its share of the variation in u left after the fixed effects and trend
     (leverage), its contribution to beta, and its residual variance;
  2. the restricted wild cluster bootstrap test of beta = 0 (null imposed; Webb weights, 9,999 draws, and
     Rademacher weights enumerated exactly, 2^10 = 1,024), which tests the same hypothesis as the
     permutation test with the year clustering of the interval;
  3. the plan's permutation test with the statistic studentised by the cluster-robust standard error
     instead of the classical one;
  4. the plan's permutation scheme with whole years permuted together (same permutation for all months);
  5. 95% intervals by inverting the permutation test (Freedman-Lane, test of beta = b0 on z - b0 u);
  6. the leave-one-year-out jackknife standard error.

  python step6d_exploratory_test1_inference.py --hindcast H.csv --uncertainty U.csv --out results/step6/exploratory
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats

import step6_link_analysis as L

SEED = 20261013


def setup(d, xcol="unc_mean_k"):
    X = L.design(d)
    Q = L.projector(X)
    z = d["z"].values.astype(float)
    x = d[xcol].values.astype(float)
    n, k = len(z), Q.shape[1] + 1
    xr = L.residualise(Q, x)
    zr = L.residualise(Q, z)
    st_codes, st_levels = pd.factorize(d["station"])
    years = np.sort(d["year"].unique())
    y_codes = np.searchsorted(years, d["year"].values)
    m_codes = d["month"].values - 1
    pos = -np.ones((len(st_levels), 12, len(years)), int)
    pos[st_codes, m_codes, y_codes] = np.arange(n)
    return dict(Q=Q, z=z, x=x, n=n, k=k, xr=xr, zr=zr, years=years, y=y_codes, m=m_codes, st=st_codes, pos=pos)


def stats_of(S, zr_perm):
    b, r = L.beta_t(S["xr"], L.residualise(S["Q"], zr_perm))
    sxx = float(S["xr"] @ S["xr"])
    t_cl = b / np.sqrt(float(r @ r) / (S["n"] - S["k"]) / sxx)
    t_cr = b / L.cluster_se(S["xr"], r, S["y"], len(S["years"]), S["n"], S["k"])
    return b, t_cl, t_cr


def perm_index(S, rng, whole_year=False):
    G = len(S["years"])
    if whole_year:
        p = rng.permutation(G)
        perm = np.tile(p, (12, 1))
    else:
        perm = np.stack([rng.permutation(G) for _ in range(12)])
    idx = S["pos"][S["st"], S["m"], perm[S["m"], S["y"]]]
    return np.where(idx < 0, np.arange(S["n"]), idx)


def permutation_tests(S, n_perm, rng, b0=0.0):
    """Freedman-Lane test of beta = b0: residuals of the reduced model for z - b0 * x are permuted."""
    zr0 = L.residualise(S["Q"], S["z"] - b0 * S["x"])
    b_obs, tcl_obs, tcr_obs = stats_of(S, zr0)
    c_cl = c_cr = c_wy = 0
    for _ in range(n_perm):
        idx = perm_index(S, rng)
        _, tcl, tcr = stats_of(S, zr0[idx])
        c_cl += abs(tcl) >= abs(tcl_obs)
        c_cr += abs(tcr) >= abs(tcr_obs)
        _, tcl_w, _ = stats_of(S, zr0[perm_index(S, rng, whole_year=True)])
        c_wy += abs(tcl_w) >= abs(tcl_obs)
    f = lambda c: (c + 1) / (n_perm + 1)
    return dict(b0=b0, t_classical=tcl_obs, t_cluster=tcr_obs, p_perm_classical=f(c_cl), p_perm_cluster_t=f(c_cr),
                p_perm_whole_year=f(c_wy))


def wcr(S, weights, rng=None, n_boot=None, enumerate_all=False):
    """Restricted wild cluster bootstrap-t test of beta = 0 (null imposed), clusters = years."""
    G = len(S["years"])
    b_obs, _, t_obs = stats_of(S, S["zr"])
    e0 = S["zr"]                                   # residuals of the model without u (null imposed)
    if enumerate_all:
        draws = np.array(list(itertools.product([-1.0, 1.0], repeat=G)))
    else:
        draws = rng.choice(weights, (n_boot, G))
    ts = np.empty(len(draws))
    for i, w in enumerate(draws):
        ystar = L.residualise(S["Q"], e0 * w[S["y"]])
        _, _, ts[i] = stats_of(S, ystar)
    return float(np.mean(np.abs(ts) >= abs(t_obs))), t_obs, len(draws)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hindcast", default="results/step2/hindcast_station_month_lstm5seeds.csv")
    ap.add_argument("--uncertainty", default="results/step1c/station_monthly_l4_uncertainty.csv")
    ap.add_argument("--n-perm", type=int, default=9999)
    ap.add_argument("--n-perm-ci", type=int, default=1999)
    ap.add_argument("--out", default="results/step6/exploratory")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    d = L.build_station_months(pd.read_csv(a.hindcast), pd.read_csv(a.uncertainty), "sarima")
    S = setup(d)

    # 1. years
    b_obs, t_cl, t_cr = stats_of(S, S["zr"])
    r_full = S["zr"] - b_obs * S["xr"]
    sxx = float(S["xr"] @ S["xr"])
    rows = []
    for g, yr in enumerate(S["years"]):
        m = S["y"] == g
        rows.append(dict(year=int(yr), leverage_share=float((S["xr"][m] ** 2).sum() / sxx),
                         contribution_to_beta_per_0p1K=float((S["xr"][m] @ S["zr"][m]) / sxx) * 0.1,
                         score=float(S["xr"][m] @ r_full[m]), resid_sd=float(r_full[m].std(ddof=1)),
                         mean_u_dev_k=float(S["xr"][m].mean())))
    years = pd.DataFrame(rows)
    years["score_share"] = years["score"] ** 2 / (years["score"] ** 2).sum()
    years.to_csv(os.path.join(a.out, "test1_years.csv"), index=False)

    # jackknife by year
    loyo = []
    for g in range(len(S["years"])):
        keep = d["year"] != S["years"][g]
        Sg = setup(d[keep].reset_index(drop=True))
        loyo.append(stats_of(Sg, Sg["zr"])[0])
    loyo = np.array(loyo)
    G = len(loyo)
    se_jk = np.sqrt((G - 1) / G * ((loyo - loyo.mean()) ** 2).sum())
    se_cr = L.cluster_se(S["xr"], r_full, S["y"], G, S["n"], S["k"])
    se_cl = np.sqrt(float(r_full @ r_full) / (S["n"] - S["k"]) / sxx)

    # 2-4. tests of beta = 0
    pt = permutation_tests(S, a.n_perm, rng)
    p_wcr_webb, _, nb = wcr(S, L.WEBB, rng, a.n_perm)
    p_wcr_rad, _, ne = wcr(S, None, enumerate_all=True)

    # 5. permutation-inverted 95% intervals (classical-t Freedman-Lane, as in the plan)
    grid = np.round(np.arange(-1.2, 0.61, 0.02), 3) / 0.1          # per K
    inv = []
    for b0 in grid:
        r = permutation_tests(S, a.n_perm_ci, np.random.default_rng(SEED + 1), b0=b0)
        inv.append(dict(b0_per_0p1K=b0 * 0.1, p_classical=r["p_perm_classical"], p_cluster_t=r["p_perm_cluster_t"],
                        p_whole_year=r["p_perm_whole_year"]))
    inv = pd.DataFrame(inv)
    inv.to_csv(os.path.join(a.out, "test1_inverted_pvalues.csv"), index=False)

    def interval(col):
        ok = inv[inv[col] > 0.05]["b0_per_0p1K"]
        return (float(ok.min()), float(ok.max())) if len(ok) else (np.nan, np.nan)

    summary = pd.DataFrame([
        dict(check="beta per 0.1 K", value=b_obs * 0.1),
        dict(check="SE classical (per 0.1 K)", value=se_cl * 0.1),
        dict(check="SE cluster-robust CR1, years (per 0.1 K)", value=se_cr * 0.1),
        dict(check="SE jackknife, leave one year out (per 0.1 K)", value=se_jk * 0.1),
        dict(check="t classical", value=pt["t_classical"]),
        dict(check="t cluster-robust", value=pt["t_cluster"]),
        dict(check="p permutation, classical t (plan scheme)", value=pt["p_perm_classical"]),
        dict(check="p permutation, cluster-robust t (plan scheme)", value=pt["p_perm_cluster_t"]),
        dict(check="p permutation, whole years, classical t", value=pt["p_perm_whole_year"]),
        dict(check=f"p restricted wild cluster bootstrap, Webb ({nb})", value=p_wcr_webb),
        dict(check=f"p restricted wild cluster bootstrap, Rademacher exact ({ne})", value=p_wcr_rad),
        dict(check="p normal, cluster-robust t", value=2 * stats.norm.sf(abs(pt["t_cluster"]))),
        dict(check="p t(9), cluster-robust t", value=2 * stats.t.sf(abs(pt["t_cluster"]), G - 1)),
        dict(check="95% interval, inverted permutation (classical t): low", value=interval("p_classical")[0]),
        dict(check="95% interval, inverted permutation (classical t): high", value=interval("p_classical")[1]),
        dict(check="95% interval, inverted permutation (cluster-robust t): low", value=interval("p_cluster_t")[0]),
        dict(check="95% interval, inverted permutation (cluster-robust t): high", value=interval("p_cluster_t")[1]),
        dict(check="95% interval, inverted whole-year permutation: low", value=interval("p_whole_year")[0]),
        dict(check="95% interval, inverted whole-year permutation: high", value=interval("p_whole_year")[1]),
        dict(check="leave-one-year-out beta min (per 0.1 K)", value=loyo.min() * 0.1),
        dict(check="leave-one-year-out beta max (per 0.1 K)", value=loyo.max() * 0.1),
    ])
    summary.to_csv(os.path.join(a.out, "test1_inference_check.csv"), index=False)
    pd.set_option("display.width", 200)
    print(summary.round(4).to_string(index=False))
    print()
    print(years.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
